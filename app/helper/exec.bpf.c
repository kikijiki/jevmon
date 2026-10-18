// SPDX-License-Identifier: GPL-2.0
/*
 * jevmon-exec BPF side: one record per successful execve.
 *
 * sched_process_exec fires only after the exec committed, so failed execve calls (ENOENT on a
 * PATH probe, EACCES) never produce an event. The resolved binary path is walked from
 * bprm->file (dentry parents, crossing mounts), like d_path: bprm->file is the file that really
 * runs (the interpreter for a script), not the execve argument, which is reported separately as
 * `filename`. bpf_d_path itself is not allowed in this program type. argv is read from the new
 * mm. Kernel types are the minimal CO-RE subsets below, so no generated vmlinux.h is needed.
 */
#include <linux/types.h>
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_tracing.h>
#include "exec.h"

#define CORE __attribute__((preserve_access_index))

typedef struct { __u32 val; } kuid_t;
typedef struct { __u32 val; } kgid_t;

struct cred {
	kuid_t uid;
	kgid_t gid;
	kuid_t suid;
	kgid_t sgid;
	kuid_t euid;
} CORE;

struct mm_struct {
	unsigned long arg_start;
	unsigned long arg_end;
} CORE;

struct qstr {
	const unsigned char *name;
} CORE;

struct dentry {
	struct dentry *d_parent;
	struct qstr d_name;
} CORE;

struct vfsmount {
	struct dentry *mnt_root;
} CORE;

struct mount {
	struct mount *mnt_parent;
	struct dentry *mnt_mountpoint;
	struct vfsmount mnt;
} CORE;

struct path {
	struct vfsmount *mnt;
	struct dentry *dentry;
} CORE;

struct file {
	struct path f_path;
} CORE;

struct linux_binprm {
	struct file *file;
	const char *filename;
	int argc;
} CORE;

struct task_struct {
	int tgid;
	struct task_struct *real_parent;
	struct task_struct *group_leader;
	struct mm_struct *mm;
	const struct cred *cred;
	__u64 start_boottime;
	char comm[16];
} CORE;

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 16 * 1024 * 1024);
} events SEC(".maps");

/* [0] = execs seen, [1] = records lost (ring buffer full). */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 2);
	__type(key, __u32);
	__type(value, __u64);
} counters SEC(".maps");

#define PATH_END 1024
#define PATH_MIN 256 /* stop walking when less room than one name is left */
#define MAX_COMPONENTS 24

struct walk_buf {
	char path[2 * PATH_END]; /* the path is built backwards ending at PATH_END */
	char name[256];
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct walk_buf);
} walk SEC(".maps");

static __always_inline void bump(__u32 idx)
{
	__u64 *v = bpf_map_lookup_elem(&counters, &idx);
	if (v)
		*v += 1;
}

/* Resolved path of bprm->file into ev->exe. Returns 0 on success. */
static __always_inline int resolve_exe(struct linux_binprm *bprm, struct exec_event *ev)
{
	__u32 zero = 0;
	struct walk_buf *w = bpf_map_lookup_elem(&walk, &zero);
	if (!w)
		return -1;
	struct file *file = BPF_CORE_READ(bprm, file);
	struct dentry *dentry = BPF_CORE_READ(file, f_path.dentry);
	struct vfsmount *vfs = BPF_CORE_READ(file, f_path.mnt);
	struct mount *mnt = (struct mount *)((char *)vfs - bpf_core_field_offset(struct mount, mnt));
	struct dentry *root = BPF_CORE_READ(vfs, mnt_root);
	__u32 pos = PATH_END;
	int done = 0;

#pragma unroll
	for (int i = 0; i < MAX_COMPONENTS; i++) {
		if (done)
			break;
		if (dentry == root) {
			struct mount *up = BPF_CORE_READ(mnt, mnt_parent);
			if (up == mnt) {
				done = 1; /* global root */
				continue;
			}
			dentry = BPF_CORE_READ(mnt, mnt_mountpoint);
			mnt = up;
			root = BPF_CORE_READ(mnt, mnt.mnt_root);
			continue;
		}
		struct dentry *parent = BPF_CORE_READ(dentry, d_parent);
		if (parent == dentry) {
			done = 1;
			continue;
		}
		const unsigned char *name = BPF_CORE_READ(dentry, d_name.name);
		long n = bpf_probe_read_kernel_str(w->name, sizeof(w->name), name);
		if (n <= 1 || n > (long)sizeof(w->name))
			return -1;
		__u32 len = (__u32)(n - 1) & 255;
		if (pos < PATH_MIN)
			return -1;
		pos -= len;
		bpf_probe_read_kernel(&w->path[pos & (2 * PATH_END - 1)], len, w->name);
		pos -= 1;
		w->path[pos & (2 * PATH_END - 1)] = '/';
		dentry = parent;
	}
	if (!done)
		return -1; /* deeper than MAX_COMPONENTS */
	if (pos == PATH_END) { /* the root itself */
		ev->exe[0] = '/';
		ev->exe[1] = 0;
		return 0;
	}
	__u32 total = PATH_END - pos;
	if (total >= EXE_MAX)
		return -1;
	total &= EXE_MAX - 1;
	bpf_probe_read_kernel(ev->exe, total, &w->path[pos & (2 * PATH_END - 1)]);
	ev->exe[total] = 0;
	return 0;
}

SEC("tp_btf/sched_process_exec")
int BPF_PROG(on_exec, struct task_struct *p, int old_pid, struct linux_binprm *bprm)
{
	bump(0);
	struct exec_event *ev = bpf_ringbuf_reserve(&events, sizeof(*ev), 0);
	if (!ev) {
		bump(1);
		return 0;
	}
	ev->flags = 0;
	ev->mono_ns = bpf_ktime_get_ns();
	ev->cgroup_id = bpf_get_current_cgroup_id();
	__u64 ug = bpf_get_current_uid_gid();
	ev->uid = (__u32)ug;
	ev->gid = ug >> 32;
	ev->tgid = BPF_CORE_READ(p, tgid);
	struct task_struct *parent = BPF_CORE_READ(p, real_parent);
	ev->ppid = BPF_CORE_READ(parent, tgid);
	struct task_struct *leader = BPF_CORE_READ(p, group_leader);
	ev->start_ns = BPF_CORE_READ(leader, start_boottime);
	ev->pstart_ns = BPF_CORE_READ(parent, group_leader, start_boottime);
	ev->euid = BPF_CORE_READ(p, cred, euid.val);
	ev->argc = BPF_CORE_READ(bprm, argc);
	bpf_get_current_comm(ev->comm, sizeof(ev->comm));
	BPF_CORE_READ_STR_INTO(&ev->pcomm, parent, comm);
	bpf_probe_read_kernel_str(ev->filename, sizeof(ev->filename), BPF_CORE_READ(bprm, filename));

	if (resolve_exe(bprm, ev)) {
		ev->exe[0] = 0;
		ev->flags |= FLAG_EXE_MISSING;
	}

	ev->argv_len = 0;
	struct mm_struct *mm = BPF_CORE_READ(p, mm);
	unsigned long start = BPF_CORE_READ(mm, arg_start);
	unsigned long end = BPF_CORE_READ(mm, arg_end);
	if (end > start) {
		__u32 len = end - start;
		if (len > ARGV_MAX) {
			len = ARGV_MAX;
			ev->flags |= FLAG_ARGV_TRUNC;
		}
		if (bpf_probe_read_user(ev->argv, len, (void *)start) == 0)
			ev->argv_len = len;
		else
			ev->flags |= FLAG_ARGV_FAIL;
	}
	bpf_ringbuf_submit(ev, 0);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
