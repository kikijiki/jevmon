/* Record shared by the BPF program and the loader. */
#ifndef JEVMON_EXEC_H
#define JEVMON_EXEC_H

#define ARGV_MAX 4096 /* bytes of argv copied per exec */
#define EXE_MAX 512   /* resolved exe path */
#define FNAME_MAX 256 /* raw execve filename */

#define FLAG_ARGV_TRUNC 1  /* argv was longer than ARGV_MAX */
#define FLAG_ARGV_FAIL 2   /* argv could not be read (page not resident) */
#define FLAG_EXE_MISSING 4 /* the kernel path walk failed (too deep or too long) */

struct exec_event {
	__u64 mono_ns;  /* CLOCK_MONOTONIC at the tracepoint */
	__u64 start_ns; /* process start, CLOCK_BOOTTIME */
	__u64 pstart_ns; /* parent start, CLOCK_BOOTTIME: tells a reused pid from the parent we know */
	__u64 cgroup_id;
	__u32 tgid;
	__u32 ppid;
	__u32 uid;
	__u32 gid;
	__u32 euid;
	__u32 argc;
	__u32 argv_len;
	__u32 flags;
	char comm[16];
	char pcomm[16];
	char exe[EXE_MAX];
	char filename[FNAME_MAX];
	char argv[ARGV_MAX];
};

#endif
