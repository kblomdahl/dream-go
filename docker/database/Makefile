.PHONY: clean staging

VERSION?=0.5.0

all: create-volume build

clean:
	rm -rf staging/

create-volume:
	docker volume create dream_db

staging:
	mkdir -p staging/
	cp ../../tools/server.py staging/server.py

build: staging
	docker build -t "dream_go/db:$(VERSION)" .

run:
	# this is for local deployments only, for a larger scale deployment you
	# will want to push the images to some repository and then deploy them from
	# there.
	# 
	# You will however still need to do the following:
	# 
	# - expose port `8080`
	# - mount some persistent storage to `/mnt/dream_db/`
	# 
	docker run --name "dream_db" \
		-p 8080:8080 -v "$(realpath ../../data):/mnt/dream_db" \
		-td --rm "dream_go/db:$(VERSION)"
