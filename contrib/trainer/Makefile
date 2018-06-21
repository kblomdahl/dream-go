# Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

VERSION?=0.5.0
DEVICE?=1
DB?=localhost:5000

.PHONY: clean docker-build docker-run docker

clean:
	rm -rf .staging/

docker-build:
	mkdir -p .staging/
	cp ../../tools/upload2rest.py .staging/upload2rest.py

	docker build -t "dream_go/trainer:$(VERSION)" .

docker-run: docker-build
	docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$(DEVICE) \
		-v "$(realpath models):/app/models" \
		-p 6006:6006 -e DB=$(DB) \
		--user 1000:1000 --net=host \
		-td --rm "dream_go/trainer:$(VERSION)"

docker: docker-run
