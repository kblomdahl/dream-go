#!/bin/sh

if [[ $# -eq 0 ]]; then
  exec "/bin/bash"
else
  exec "$@"
fi

