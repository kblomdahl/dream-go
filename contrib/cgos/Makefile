.PHONY: all clean

all: .staging/dream_go .staging/dream_go.json .cgos_password
	docker-compose build \
	    --build-arg GIT_REV=`git show-ref --head --tags --hash=6 HEAD`.`sha1sum < .staging/dream_go.json | head -c 6` \
	    --build-arg CGOS_PASSWORD=`cat .cgos_password`

clean:
	docker-compose rm -f
	rm -rf .staging

.staging/dream_go:
	cargo build --locked --release -Z unstable-options --out-dir .staging

.staging/dream_go.json: ../../dream_go.json
	cp $< $@

.cgos_password:
	test -f .cgos_password || \
	    (head -c 1024 < /dev/urandom | sha1sum -b - | cut -d ' ' -f 1 | head -c 16 > .cgos_password)
