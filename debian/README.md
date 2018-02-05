# Distributable packages

These package will install the following files, it will not install any
dependencies beyond this because there are no universal packages for CUDA and
cuDNN:

* `/usr/games/dream_go`
* `/usr/share/dreamgo/dream_go.json`

## Minimal package

This will build the archive `dreamgo-X_amd64.tar.gz` that contains the necessary
binaries and weights in a flat archive.

```bash
fakeroot make
```

## Debian / Ubuntu packages

This will build a package that can be installed on Debian or Ubuntu systems.

```bash
fakeroot make
```

## Fedora / Redhat packages

This will build a package that can be installed on Fedora or Redhat systems.

```bash
fakeroot make
fakeroot alien -r dreamgo-*.deb
```
