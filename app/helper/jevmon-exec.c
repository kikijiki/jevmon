// SPDX-License-Identifier: GPL-2.0
/*
 * jevmon-exec: stream every successful execve to a unix socket as newline-delimited JSON.
 *
 * Runs as root (CAP_BPF + CAP_PERFMON + CAP_SYS_RESOURCE). It loads exec.bpf.c, reads the ring
 * buffer and writes lines to the clients that connect to --socket. It accepts no commands from
 * clients and never reads from them. A client is served only when SO_PEERCRED says its uid is
 * one of the --uid values; everyone else is closed without a byte.
 *
 * Lines (all JSON objects, one per line):
 *   {"t":"hello","v":1,...}   once per connection
 *   {"t":"exec",...}          one per exec
 *   {"t":"hb",...}            every second: exec counters, including events lost
 *
 * --synthetic RATE replaces the BPF source with generated records (no privileges, no BPF): it
 * exists so the socket, encoder and the daemon can be tested end to end without root.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <linux/types.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#include "exec.h"

#define VERSION "0.1.0"
#define PROTO 1
#define ARGS_MAX 64
#define MAX_CLIENTS 8
#define MAX_UIDS 8
#define CLIENT_BUF_MAX (16u << 20)

#ifdef BPF_OBJ
__asm__(".section .rodata\n.balign 8\n.global bpf_obj_start\nbpf_obj_start:\n"
	".incbin \"" BPF_OBJ "\"\n.global bpf_obj_end\nbpf_obj_end:\n.byte 0\n.previous\n");
extern const char bpf_obj_start[], bpf_obj_end[];
#endif

/* ---- growable byte buffer ---- */
struct sbuf {
	char *p;
	size_t len, cap;
};

static int sb_reserve(struct sbuf *b, size_t extra)
{
	if (b->len + extra <= b->cap)
		return 0;
	size_t cap = b->cap ? b->cap : 1024;
	while (cap < b->len + extra)
		cap *= 2;
	char *n = realloc(b->p, cap);
	if (!n)
		return -1;
	b->p = n;
	b->cap = cap;
	return 0;
}

static void sb_put(struct sbuf *b, const char *s, size_t n)
{
	if (sb_reserve(b, n))
		return;
	memcpy(b->p + b->len, s, n);
	b->len += n;
}

static void sb_printf(struct sbuf *b, const char *fmt, ...)
{
	va_list ap;
	char tmp[512];
	va_start(ap, fmt);
	int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
	va_end(ap);
	if (n > 0)
		sb_put(b, tmp, (size_t)n < sizeof(tmp) ? (size_t)n : sizeof(tmp) - 1);
}

/* JSON string from bytes. Invalid UTF-8 becomes U+FFFD so every line parses. */
static void sb_json_str(struct sbuf *b, const unsigned char *s, size_t n)
{
	sb_put(b, "\"", 1);
	for (size_t i = 0; i < n;) {
		unsigned char c = s[i];
		if (c == '"' || c == '\\') {
			char e[2] = { '\\', (char)c };
			sb_put(b, e, 2);
			i++;
		} else if (c < 0x20) {
			sb_printf(b, "\\u%04x", c);
			i++;
		} else if (c < 0x80) {
			sb_put(b, (const char *)&s[i], 1);
			i++;
		} else {
			size_t need = c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc2 ? 2 : 0;
			int ok = need && c <= 0xf4 && i + need <= n;
			for (size_t k = 1; ok && k < need; k++)
				ok = (s[i + k] & 0xc0) == 0x80;
			if (ok && need == 3) {
				unsigned cp = ((c & 0x0f) << 12) | ((s[i + 1] & 0x3f) << 6);
				ok = cp >= 0x800 && !(cp >= 0xd800 && cp < 0xe000);
			} else if (ok && need == 4) {
				unsigned cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3f) << 12);
				ok = cp >= 0x10000 && cp < 0x110000;
			}
			if (ok) {
				sb_put(b, (const char *)&s[i], need);
				i += need;
			} else {
				sb_put(b, "\xef\xbf\xbd", 3);
				i++;
			}
		}
	}
	sb_put(b, "\"", 1);
}

/* ---- state ---- */
struct client {
	int fd;
	struct sbuf out;
	size_t sent;
	__u64 dropped;
};

static struct client clients[MAX_CLIENTS];
static int nclients;
static uid_t allowed_uids[MAX_UIDS];
static int nuids;
static volatile sig_atomic_t stop_flag;
static int listen_fd = -1;
static FILE *dump;
static const char *exe_source = "kernel";
static int synthetic_mode;

static __u64 seq_emitted, undelivered, client_dropped_total, truncated_count, exe_missing_count;
static __u64 last_seen, last_lost, last_emitted;
static __u64 sec_count, peak_per_s, argv_bytes_total;
static double started_wall;

static double now_real(void)
{
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return t.tv_sec + t.tv_nsec / 1e9;
}

static __u64 now_mono_ns(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (__u64)t.tv_sec * 1000000000ull + t.tv_nsec;
}

static void on_signal(int sig)
{
	(void)sig;
	stop_flag = 1;
}

static void client_close(int i)
{
	close(clients[i].fd);
	free(clients[i].out.p);
	clients[i] = clients[--nclients];
	memset(&clients[nclients], 0, sizeof(clients[nclients]));
}

/* Queue one complete line to every client. A client that is too far behind loses the line. */
static void broadcast(const struct sbuf *line, int count_events)
{
	if (dump)
		fwrite(line->p, 1, line->len, dump);
	if (count_events && nclients == 0)
		undelivered++;
	for (int i = 0; i < nclients; i++) {
		struct client *c = &clients[i];
		if (c->out.len - c->sent + line->len > CLIENT_BUF_MAX) {
			if (count_events) {
				c->dropped++;
				client_dropped_total++;
			}
			continue;
		}
		sb_put(&c->out, line->p, line->len);
	}
}

static void encode_exec(struct sbuf *b, const struct exec_event *ev_in)
{
	/* exe_src: 0 = kernel path walk of the file that runs, 1 = realpath() of the execve
	 * filename (the walk failed), 2 = unresolved */
	struct exec_event ev_copy;
	const struct exec_event *ev = ev_in;
	int exe_src = 0;
	if (ev->flags & FLAG_EXE_MISSING) {
		exe_src = 2;
		if (ev->filename[0] == '/') {
			char resolved[PATH_MAX];
			if (realpath(ev->filename, resolved) && strlen(resolved) < EXE_MAX) {
				memcpy(&ev_copy, ev, offsetof(struct exec_event, argv));
				strcpy(ev_copy.exe, resolved);
				memcpy(ev_copy.argv, ev->argv, ev->argv_len);
				ev = &ev_copy;
				exe_src = 1;
			}
		}
	}
	double wall = now_real() - (double)(now_mono_ns() - ev->mono_ns) / 1e9;
	const char *argv = ev->argv;
	size_t alen = ev->argv_len;
	int trunc = (ev->flags & FLAG_ARGV_TRUNC) != 0;

	sb_printf(b,
		  "{\"t\":\"exec\",\"seq\":%llu,\"ts\":%.6f,\"mono\":%llu,\"start\":%llu,"
		  "\"pid\":%u,\"ppid\":%u,\"pstart\":%llu,\"uid\":%u,\"euid\":%u,\"gid\":%u,\"cgroup\":%llu,"
		  "\"argc\":%u,\"flags\":%u,\"exe_src\":%d,\"comm\":",
		  (unsigned long long)++seq_emitted, wall, (unsigned long long)ev->mono_ns,
		  (unsigned long long)ev->start_ns, ev->tgid, ev->ppid, (unsigned long long)ev->pstart_ns, ev->uid, ev->euid, ev->gid,
		  (unsigned long long)ev->cgroup_id, ev->argc, ev->flags, exe_src);
	sb_json_str(b, (const unsigned char *)ev->comm, strnlen(ev->comm, sizeof(ev->comm)));
	sb_put(b, ",\"pcomm\":", 9);
	sb_json_str(b, (const unsigned char *)ev->pcomm, strnlen(ev->pcomm, sizeof(ev->pcomm)));
	sb_put(b, ",\"exe\":", 7);
	sb_json_str(b, (const unsigned char *)ev->exe, strnlen(ev->exe, sizeof(ev->exe)));
	sb_put(b, ",\"filename\":", 12);
	sb_json_str(b, (const unsigned char *)ev->filename, strnlen(ev->filename, sizeof(ev->filename)));
	sb_put(b, ",\"argv\":[", 9);
	unsigned n = 0;
	size_t i = 0;
	while (i < alen) {
		if (n == ARGS_MAX) {
			trunc = 1;
			break;
		}
		size_t end = i;
		while (end < alen && argv[end] != 0)
			end++;
		if (end == alen && !(ev->flags & FLAG_ARGV_TRUNC) && argv[alen - 1] != 0) {
			/* no terminator and not flagged: still emit what is there */
		}
		if (n)
			sb_put(b, ",", 1);
		sb_json_str(b, (const unsigned char *)argv + i, end - i);
		n++;
		i = end + 1;
	}
	if (n < ev->argc && !trunc)
		trunc = 1; /* fewer args read than the kernel counted */
	sb_printf(b, "],\"trunc\":%d}\n", trunc);
	if (trunc)
		truncated_count++;
	if (exe_src == 2)
		exe_missing_count++;
	argv_bytes_total += alen;
}

static void emit_exec(const struct exec_event *ev)
{
	struct sbuf line = { 0 };
	encode_exec(&line, ev);
	broadcast(&line, 1);
	free(line.p);
	sec_count++;
}

/* ---- BPF side ---- */
static struct bpf_object *obj;
static struct ring_buffer *rb;
static int counters_fd = -1;

static int on_record(void *ctx, void *data, size_t size)
{
	(void)ctx;
	if (size >= sizeof(struct exec_event))
		emit_exec(data);
	return 0;
}

static int libbpf_print(enum libbpf_print_level lvl, const char *fmt, va_list ap)
{
	if (lvl == LIBBPF_DEBUG)
		return 0;
	return vfprintf(stderr, fmt, ap);
}

static int bpf_start(void)
{
#ifdef BPF_OBJ
	libbpf_set_print(libbpf_print);
	obj = bpf_object__open_mem(bpf_obj_start, bpf_obj_end - bpf_obj_start, NULL);
	if (!obj) {
		fprintf(stderr, "jevmon-exec: cannot open embedded BPF object\n");
		return -1;
	}
	if (bpf_object__load(obj)) {
		fprintf(stderr, "jevmon-exec: BPF load failed\n");
		return -1;
	}
	struct bpf_program *prog;
	bpf_object__for_each_program(prog, obj) {
		if (!bpf_program__attach(prog)) {
			fprintf(stderr, "jevmon-exec: attach %s failed: %s\n", bpf_program__name(prog),
				strerror(errno));
			return -1;
		}
	}
	struct bpf_map *m = bpf_object__find_map_by_name(obj, "events");
	struct bpf_map *c = bpf_object__find_map_by_name(obj, "counters");
	if (!m || !c)
		return -1;
	counters_fd = bpf_map__fd(c);
	rb = ring_buffer__new(bpf_map__fd(m), on_record, NULL, NULL);
	return rb ? 0 : -1;
#else
	fprintf(stderr, "jevmon-exec: built without the BPF object; use --synthetic\n");
	return -1;
#endif
}

static void read_counters(__u64 *seen, __u64 *lost)
{
	*seen = *lost = 0;
	if (counters_fd < 0)
		return;
	int ncpu = libbpf_num_possible_cpus();
	if (ncpu <= 0)
		return;
	__u64 *vals = calloc((size_t)ncpu, sizeof(*vals));
	if (!vals)
		return;
	for (__u32 k = 0; k < 2; k++) {
		if (bpf_map_lookup_elem(counters_fd, &k, vals) == 0) {
			__u64 sum = 0;
			for (int i = 0; i < ncpu; i++)
				sum += vals[i];
			if (k == 0)
				*seen = sum;
			else
				*lost = sum;
		}
	}
	free(vals);
}

/* ---- synthetic source ---- */
static void synth_one(unsigned long n)
{
	static const char *const exes[] = { "/usr/bin/true", "/usr/bin/ls", "/usr/bin/git",
					    "/usr/bin/python3", "/usr/bin/bash", "/usr/bin/sleep" };
	struct exec_event *ev = calloc(1, sizeof(*ev));
	if (!ev)
		return;
	const char *exe = exes[n % 6];
	ev->mono_ns = now_mono_ns();
	ev->start_ns = ev->mono_ns;
	ev->cgroup_id = 1000 + n % 7;
	ev->tgid = 100000 + (unsigned)(n % 400000);
	ev->ppid = 1 + (unsigned)(n % 50);
	ev->uid = ev->euid = ev->gid = 1000;
	snprintf(ev->comm, sizeof(ev->comm), "synth%lu", n % 6);
	snprintf(ev->pcomm, sizeof(ev->pcomm), "parent");
	snprintf(ev->exe, sizeof(ev->exe), "%s", exe);
	snprintf(ev->filename, sizeof(ev->filename), "%s", exe);
	int len = snprintf(ev->argv, sizeof(ev->argv), "%s%c--n%c%lu%c", exe, 0, 0, n, 0);
	ev->argv_len = (unsigned)len;
	ev->argc = 3;
	emit_exec(ev);
	free(ev);
}

/* ---- sockets ---- */
static int listen_start(const char *path)
{
	struct sockaddr_un a = { .sun_family = AF_UNIX };
	if (strlen(path) >= sizeof(a.sun_path)) {
		fprintf(stderr, "jevmon-exec: socket path too long\n");
		return -1;
	}
	strcpy(a.sun_path, path);
	int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
	if (fd < 0)
		return -1;
	unlink(path);
	mode_t old = umask(0);
	int rc = bind(fd, (struct sockaddr *)&a, sizeof(a));
	umask(old);
	if (rc || listen(fd, 8)) {
		fprintf(stderr, "jevmon-exec: cannot listen on %s: %s\n", path, strerror(errno));
		close(fd);
		return -1;
	}
	/* mode 0666: the gate is SO_PEERCRED in accept_clients, not the file mode */
	chmod(path, 0666);
	return fd;
}

static int uid_allowed(uid_t u)
{
	for (int i = 0; i < nuids; i++)
		if (allowed_uids[i] == u)
			return 1;
	return 0;
}

static void accept_clients(void)
{
	for (;;) {
		int fd = accept4(listen_fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
		if (fd < 0)
			return;
		struct ucred cr;
		socklen_t len = sizeof(cr);
		if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cr, &len) || !uid_allowed(cr.uid) ||
		    nclients >= MAX_CLIENTS) {
			close(fd);
			continue;
		}
		struct client *c = &clients[nclients++];
		memset(c, 0, sizeof(*c));
		c->fd = fd;
		sb_printf(&c->out,
			  "{\"t\":\"hello\",\"v\":%d,\"helper\":\"jevmon-exec\",\"version\":\"%s\","
			  "\"argv_max\":%d,\"args_max\":%d,\"exe_source\":\"%s\","
			  "\"synthetic\":%s,\"ts\":%.6f}\n",
			  PROTO, VERSION, ARGV_MAX, ARGS_MAX, exe_source,
			  synthetic_mode ? "true" : "false", now_real());
	}
}

static void flush_clients(void)
{
	for (int i = nclients - 1; i >= 0; i--) {
		struct client *c = &clients[i];
		while (c->sent < c->out.len) {
			ssize_t n = send(c->fd, c->out.p + c->sent, c->out.len - c->sent,
					 MSG_NOSIGNAL | MSG_DONTWAIT);
			if (n > 0) {
				c->sent += (size_t)n;
			} else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
				break;
			} else {
				client_close(i);
				c = NULL;
				break;
			}
		}
		if (c && c->sent == c->out.len) {
			c->sent = c->out.len = 0;
		} else if (c && c->sent > (1u << 20)) {
			memmove(c->out.p, c->out.p + c->sent, c->out.len - c->sent);
			c->out.len -= c->sent;
			c->sent = 0;
		}
	}
}

/* Clients never send; anything readable is EOF or noise, so a readable client is closed. */
static void reap_clients(const struct pollfd *pfds, int base)
{
	for (int i = nclients - 1; i >= 0; i--)
		if (pfds[base + i].revents & (POLLIN | POLLHUP | POLLERR))
			client_close(i);
}

static void heartbeat(void)
{
	__u64 seen, lost;
	read_counters(&seen, &lost);
	if (synthetic_mode)
		seen = seq_emitted;
	if (sec_count > peak_per_s)
		peak_per_s = sec_count;
	sec_count = 0;
	last_seen = seen;
	last_lost = lost;
	last_emitted = seq_emitted;
	struct sbuf line = { 0 };
	sb_printf(&line,
		  "{\"t\":\"hb\",\"ts\":%.6f,\"seen\":%llu,\"emitted\":%llu,\"lost\":%llu,"
		  "\"undelivered\":%llu,\"client_dropped\":%llu,\"clients\":%d}\n",
		  now_real(), (unsigned long long)seen, (unsigned long long)seq_emitted,
		  (unsigned long long)lost, (unsigned long long)undelivered,
		  (unsigned long long)client_dropped_total, nclients);
	broadcast(&line, 0);
	free(line.p);
}

static void summary(void)
{
	double dur = now_real() - started_wall;
	fprintf(stderr,
		"jevmon-exec summary: %.1fs  seen %llu  emitted %llu  lost %llu  undelivered %llu  "
		"client_dropped %llu\n"
		"  rate avg %.1f/s  peak %llu/s  argv avg %.0f B  truncated %llu  exe unresolved %llu\n",
		dur, (unsigned long long)last_seen, (unsigned long long)last_emitted,
		(unsigned long long)last_lost, (unsigned long long)undelivered,
		(unsigned long long)client_dropped_total, dur > 0 ? last_emitted / dur : 0.0,
		(unsigned long long)peak_per_s,
		last_emitted ? (double)argv_bytes_total / last_emitted : 0.0,
		(unsigned long long)truncated_count, (unsigned long long)exe_missing_count);
}

static void usage(void)
{
	fputs("usage: jevmon-exec --uid N [--uid N ...] [--socket PATH] [--duration SEC]\n"
	      "                   [--dump FILE] [--synthetic RATE]\n"
	      "  --uid N        uid allowed to connect (required unless --socket is empty)\n"
	      "  --socket PATH  listening socket (default /run/jevmon/exec.sock; '' = none)\n"
	      "  --duration S   exit after S seconds and print a summary\n"
	      "  --dump FILE    also append every line to FILE\n"
	      "  --synthetic R  generate R fake execs/s instead of attaching BPF (testing)\n",
	      stderr);
}

int main(int argc, char **argv)
{
	const char *sock_path = "/run/jevmon/exec.sock";
	const char *dump_path = NULL;
	double duration = 0, synth_rate = 0;

	for (int i = 1; i < argc; i++) {
		const char *a = argv[i];
		const char *v = i + 1 < argc ? argv[i + 1] : NULL;
		if (!strcmp(a, "--uid") && v && nuids < MAX_UIDS) {
			allowed_uids[nuids++] = (uid_t)strtoul(v, NULL, 10);
			i++;
		} else if (!strcmp(a, "--socket") && v) {
			sock_path = v;
			i++;
		} else if (!strcmp(a, "--duration") && v) {
			duration = atof(v);
			i++;
		} else if (!strcmp(a, "--dump") && v) {
			dump_path = v;
			i++;
		} else if (!strcmp(a, "--synthetic") && v) {
			synth_rate = atof(v);
			synthetic_mode = 1;
			i++;
		} else {
			usage();
			return 2;
		}
	}
	if (*sock_path && !nuids) {
		usage();
		return 2;
	}
	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);
	signal(SIGPIPE, SIG_IGN);
	started_wall = now_real();

	if (dump_path) {
		dump = fopen(dump_path, "a");
		if (!dump) {
			perror("jevmon-exec: dump");
			return 1;
		}
		setvbuf(dump, NULL, _IOFBF, 1 << 16);
		/* hand the file to the first allowed uid: the dump holds command lines */
		if (nuids && fchown(fileno(dump), allowed_uids[0], (gid_t)-1) && errno != EPERM)
			perror("jevmon-exec: chown dump");
		fchmod(fileno(dump), 0600);
	}
	if (*sock_path) {
		listen_fd = listen_start(sock_path);
		if (listen_fd < 0)
			return 1;
	}
	if (synthetic_mode) {
		exe_source = "kernel";
	} else if (bpf_start()) {
		return 1;
	}
	fprintf(stderr, "jevmon-exec %s: %s, exe path from %s, socket %s\n", VERSION,
		synthetic_mode ? "synthetic source" : "attached", exe_source,
		*sock_path ? sock_path : "(none)");

	double next_hb = now_real() + 1.0;
	double end = duration > 0 ? started_wall + duration : 0;
	double synth_due = now_real();
	unsigned long synth_n = 0;

	while (!stop_flag) {
		struct pollfd pfds[2 + MAX_CLIENTS];
		int np = 0;
		int rb_idx = -1, lis_idx = -1;
		if (rb) {
			rb_idx = np;
			pfds[np++] = (struct pollfd){ .fd = ring_buffer__epoll_fd(rb), .events = POLLIN };
		}
		if (listen_fd >= 0) {
			lis_idx = np;
			pfds[np++] = (struct pollfd){ .fd = listen_fd, .events = POLLIN };
		}
		int base = np;
		for (int i = 0; i < nclients; i++)
			pfds[np++] = (struct pollfd){
				.fd = clients[i].fd,
				.events = POLLIN | (clients[i].sent < clients[i].out.len ? POLLOUT : 0),
			};
		double now = now_real();
		double wake = next_hb;
		if (synth_rate > 0 && synth_due < wake)
			wake = synth_due;
		int timeout = (int)((wake - now) * 1000.0) + 1;
		if (timeout < 0)
			timeout = 0;
		if (poll(pfds, np, timeout) < 0 && errno != EINTR)
			break;
		if (rb_idx >= 0 && (pfds[rb_idx].revents & POLLIN))
			ring_buffer__consume(rb);
		if (lis_idx >= 0 && (pfds[lis_idx].revents & POLLIN))
			accept_clients();
		else
			reap_clients(pfds, base);
		now = now_real();
		if (synth_rate > 0) {
			int burst = 0;
			while (synth_due <= now && burst++ < 5000) {
				synth_one(synth_n++);
				synth_due += 1.0 / synth_rate;
			}
			if (synth_due < now - 1.0)
				synth_due = now;
		}
		if (now >= next_hb) {
			heartbeat();
			next_hb += 1.0;
			if (next_hb < now)
				next_hb = now + 1.0;
		}
		flush_clients();
		if (end && now >= end)
			break;
	}
	if (rb)
		ring_buffer__consume(rb);
	heartbeat();
	flush_clients();
	summary();
	if (dump)
		fclose(dump);
	if (*sock_path)
		unlink(sock_path);
	return 0;
}
