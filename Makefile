CONTROLLER_GEN = go run sigs.k8s.io/controller-tools/cmd/controller-gen@v0.17.3

.PHONY: fmt
fmt:
	go fmt ./...

.PHONY: generate
generate:
	$(CONTROLLER_GEN) object paths="./..."

.PHONY: manifests
manifests:
	$(CONTROLLER_GEN) crd:crdVersions=v1 webhook paths="./api/..." output:crd:artifacts:config=config/crd/bases output:webhook:artifacts:config=config/webhook

.PHONY: test
test: generate
	go test ./...

.PHONY: build
build:
	go build ./...

.PHONY: training-test
training-test:
	PYTHONPATH=. python3 -m unittest discover -s training/tests -p "test_*.py"
