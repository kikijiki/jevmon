{
  description = "jevmon - resource and process monitor driven by a small open-weight decision model";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    {
      # throwaway VM for verifying the helper's BPF program (see app/helper/vmtest.nix)
      nixosConfigurations.exec-vmtest = nixpkgs.lib.nixosSystem {
        system = "x86_64-linux";
        modules = [
          "${nixpkgs}/nixos/modules/virtualisation/qemu-vm.nix"
          ./app/helper/vmtest.nix
          { _module.args.helper = self.packages.x86_64-linux.jevmon-exec; system.stateVersion = "26.05"; }
        ];
      };
    } //
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        # The privileged exec helper (app/helper/): a CO-RE BPF program plus its loader. Built
        # with `just exec-build`; the dev shell does not need clang.
        packages.jevmon-exec = pkgs.stdenv.mkDerivation {
          pname = "jevmon-exec";
          version = "0.1.0";
          src = ./app/helper;
          nativeBuildInputs = [ pkgs.llvmPackages.clang-unwrapped ];
          buildInputs = [ pkgs.libbpf pkgs.elfutils pkgs.zlib ];
          hardeningDisable = [ "all" ];
          buildPhase = ''
            clang -O2 -g -target bpf -D__TARGET_ARCH_x86 \
              -I${pkgs.libbpf}/include -I${pkgs.linuxHeaders}/include -I. \
              -c exec.bpf.c -o exec.bpf.o
            cc -O2 -Wall -Wextra -DBPF_OBJ='"exec.bpf.o"' -I. \
              jevmon-exec.c -o jevmon-exec -lbpf -lelf -lz
          '';
          installPhase = ''
            install -Dm755 jevmon-exec $out/bin/jevmon-exec
            install -Dm644 exec.bpf.o $out/share/jevmon-exec/exec.bpf.o
          '';
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.python313
            pkgs.uv        # env + lockfile; torch comes from PyPI wheels, not nixpkgs
            pkgs.just      # task runner - recipes in ./justfile
            pkgs.ruff
            pkgs.jq
            pkgs.git
            pkgs.nixfmt
            # `just screenshots` runs the TUI in a fixed-geometry headless pane and rasterises
            # the capture with Pillow; this pins the monospace font so re-runs match.
            pkgs.tmux
            pkgs.dejavu_fonts

            # Rust TUI (app/tui/, binary jevmon-tui): build, lint, format.
            pkgs.rustc
            pkgs.cargo
            pkgs.clippy
            pkgs.rustfmt
            pkgs.rust-analyzer
          ];

          # PyPI wheels (torch) are linked against the usual libs and
          # find none of them on NixOS without this. zstd/numactls/elfutils/pciutils/
          # libdrm/libffi are what the PyPI ROCm torch wheel (`just gpu-sync`) needs at
          # import time on top of the CPU wheel's set; harmless when it is not installed.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.zstd
            pkgs.xz
            pkgs.bzip2
            pkgs.numactl
            pkgs.elfutils
            pkgs.pciutils
            pkgs.libdrm
            pkgs.libffi
          ];

          # Use the nix Python; never let uv download its own interpreter.
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = "${pkgs.python313}/bin/python3";

          # The font `app/scripts/screenshots.py` rasterises docs/screenshots/*.png with.
          JEVMON_MONO_FONT = "${pkgs.dejavu_fonts}/share/fonts/truetype/DejaVuSansMono.ttf";
        };
      });
}
