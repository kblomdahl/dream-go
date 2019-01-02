# CGOS (Computer Go Server) script

This directory contains a docker image for running _Dream Go_ on the Computer Go Server.

## Usage

You will need to create the `.cgos_password` file, and set the password for your account there. You may also want to change the username in `config.txt`.

Once you have configured your username and password, build the docker image using the make file, and start the docker container:

```bash
make && docker-compose start
```