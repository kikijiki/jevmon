# A throwaway NixOS VM that loads the helper's BPF program in a guest kernel, so the verifier,
# the tracepoint and the path walk can be checked without root on the host. Run it with
# `just exec-vmtest`; it needs /dev/kvm. The guest runs a few commands, the helper records them
# to the shared directory, and the VM powers off.
{ pkgs, helper, ... }:
{
  boot.kernelPackages = pkgs.linuxPackages_latest;
  virtualisation = {
    memorySize = 2048;
    cores = 2;
    graphics = false;
  };
  fileSystems."/mnt/out" = {
    device = "out";
    fsType = "9p";
    options = [
      "trans=virtio"
      "version=9p2000.L"
      "nofail"
    ];
  };
  users.users.root.initialHashedPassword = "";
  systemd.services.exec-vmtest = {
    wantedBy = [ "multi-user.target" ];
    path = [
      pkgs.coreutils
      pkgs.bash
      pkgs.util-linux
      pkgs.systemd
    ];
    serviceConfig = {
      Type = "oneshot";
      StandardOutput = "journal+console";
      StandardError = "journal+console";
    };
    script = ''
      set -x
      mkdir -p /mnt/out; mountpoint -q /mnt/out || mount -t 9p -o trans=virtio,version=9p2000.L out /mnt/out
      echo "guest kernel $(uname -r)" > /mnt/out/guest.txt
      ${helper}/bin/jevmon-exec --uid 0 --socket "" --duration 8 --dump /mnt/out/dump.ndjson 2> /mnt/out/helper.err &
      sleep 3
      set +e
      mkdir -p /tmp/bin; ln -s ${pkgs.coreutils}/bin/ls /tmp/bin/ls
      /tmp/bin/ls / > /dev/null
      /tmp/does-not-exist
      printf '#!${pkgs.bash}/bin/bash\necho script\n' > /tmp/s.sh; chmod +x /tmp/s.sh; /tmp/s.sh
      bash -c 'exec ${pkgs.coreutils}/bin/sleep 0.1' x1 "$(head -c 6000 /dev/zero | tr '\0' a)"
      ${pkgs.coreutils}/bin/echo $(seq 1 100) > /dev/null
      PATH=/nonexistent:/tmp/bin:$PATH ls / > /dev/null
      wait || true
      echo done >> /mnt/out/guest.txt
      sync
      systemctl poweroff
    '';
  };
}
