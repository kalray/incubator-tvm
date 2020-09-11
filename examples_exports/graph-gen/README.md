# Graph generation scripts

## Installation instructions

### 1. Install opam with
```
sh <(curl -sL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)
```

### 2. Install system dependencies
Replace with the package manager for your distribution
```
sudo <pkg-manager> install -y bubblewrap mercurial darcs ocaml
```

### 3. Install ocaml dependencies
```
opam install yojson
```

## Usage instructions

```
./graph-gen.sh <file.dot>
```
Generates a graphml file in the current folder with a name automatically given
by `mktemp` command.


## Visualization
To visualize the graphml file you will need [yEd Graph
Editor](https://www.yworks.com/products/yed) in your system.  Once you open the
graphml, go to `Tools -> Fit Node to Label` and then `Layout -> Hierarchical`.
