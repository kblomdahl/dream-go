# Distributable packages

These package will install the following files, it will not install any
dependencies beyond this because there are no universal packages for CUDA and
cuDNN:

* `/usr/games/dream_go`
* `/usr/share/dreamgo/dream_go.json`

All the packages are binary only, and has the following build dependencies:

* `alien`
* `cargo` (nightly)
* `cuda`
* `cudnn`
* `dpkg`
* `fakeroot`
* `make`

## Minimal package

This will build the archive `dreamgo-X.Y.Z_amd64.tar.gz` that contains the
necessary binaries and weights to run Dream Go in a flat archive:

```bash
fakeroot make
```

## Debian / Ubuntu packages

This will build a package that is suitable for installation on Debian and Ubuntu
systems:

```bash
fakeroot make
```

## Fedora / Redhat packages

This will build a package that is suitable for installation on Fedora and Redhat
systems:

```bash
fakeroot make
fakeroot alien -r dreamgo-*.deb
```

## Windows packages

This will require a bash prompt to build, you can acquire one from the
[MSYS2](https://www.msys2.org/) project. It will build the zip archive
`dreamgo-X.Y.Z_amd64.zip` containing the relevant files.

```bash
make zip
```

You should manually add the CUDA and cuDNN libraries you built the binary
against since there is no way to automatically detect them.
