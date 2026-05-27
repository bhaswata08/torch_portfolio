{
  description = "Cuda dev env";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };
  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;   # required for CUDA
        cudaSupport = true;
      };
    };
  in {
    devShells.${system}.default = (pkgs.buildFHSEnv {
      name = "cuda-env";

      targetPkgs = pkgs: with pkgs; [
        git
        gitRepo
        gnupg
        autoconf
        curl
        procps
        gnumake
        util-linux
        m4
        gperf
        unzip
        cudatoolkit
        linuxPackages.nvidia_x11
        libGLU libGL
        libXi libXmu freeglut
        libxext libx11 libxv libxrandr zlib
        ncurses5
        stdenv.cc
        binutils

        neovim
        nodejs
        python3
        python3Packages.pip
        gcc
        clang-tools
      ];

      multiPkgs = pkgs: with pkgs; [ zlib ];

      profile = ''
        export CUDA_PATH=${pkgs.cudatoolkit}
        export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        export EXTRA_CCFLAGS="-I/usr/include"
        export SSL_CERT_FILE="./.venv/lib/python3.14/site-packages/certifi/cacert.pem"
        export XDG_CONFIG_HOME=$HOME/.config
        export XDG_DATA_HOME=$HOME/.local/share
      '';

      runScript = "bash --init-file <(echo 'echo \"cuda-env ready\"')";
    }).env;
  };
}
