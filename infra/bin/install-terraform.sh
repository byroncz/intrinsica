#!/usr/bin/env bash
# Instala Terraform en ~/.local/bin sin sudo, verificando el SHA256 publicado
# por HashiCorp. Idempotente: si ya está la versión fijada, no hace nada.
set -euo pipefail

TERRAFORM_VERSION="1.16.4"

BIN_DIR="${HOME}/.local/bin"
BASE_URL="https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}"

case "$(uname -m)" in
  x86_64)  arch=amd64 ;;
  aarch64 | arm64) arch=arm64 ;;
  *) echo "Arquitectura no soportada: $(uname -m)" >&2; exit 1 ;;
esac

zip="terraform_${TERRAFORM_VERSION}_linux_${arch}.zip"

if [ -x "${BIN_DIR}/terraform" ] &&
   "${BIN_DIR}/terraform" version 2>/dev/null | head -n1 | grep -qx "Terraform v${TERRAFORM_VERSION}"; then
  echo "Terraform v${TERRAFORM_VERSION} ya está instalado."
  "${BIN_DIR}/terraform" version
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fsSL -o "${tmp}/${zip}" "${BASE_URL}/${zip}"
curl -fsSL -o "${tmp}/SHA256SUMS" "${BASE_URL}/terraform_${TERRAFORM_VERSION}_SHA256SUMS"

(cd "$tmp" && grep " ${zip}\$" SHA256SUMS | sha256sum -c -)

mkdir -p "$BIN_DIR"
unzip -oq "${tmp}/${zip}" terraform -d "$tmp"
install -m 0755 "${tmp}/terraform" "${BIN_DIR}/terraform"

"${BIN_DIR}/terraform" version
