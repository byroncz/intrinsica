# scaffold/layer_template/

Plantilla para crear una capa nueva. Definido en el
[TRD maestro §8.1](../../docs/TRD/plataforma_directional_change.md#81-estrategia-de-repositorio-monorepo-políglota)
y §8.6 (playbook). Se extrajo de `layers/l1_ingest`, la primera capa real.

## Regla (ADR-07)

**Una imagen por capa, modo por parámetro.** Los modos (por ejemplo
`backfill`, `daily`, `monthly-close`) se eligen al ejecutar, no con imágenes
distintas.

## Árbol de una capa

```
l<N>_<nombre>/
├── Dockerfile               # multi-stage, base slim, contexto = raíz del workspace
├── Dockerfile.dockerignore  # BuildKit lo lee junto al Dockerfile (no un .dockerignore)
├── pyproject.toml
├── VERSION                  # versión semántica de la capa
├── smoke.sh                 # prueba de humo de la imagen, la corre el CI
├── src/l<N>_<nombre>/       # __init__.py y __main__.py de arranque
├── tests/
└── config/                  # YAML versionados
```

La plantilla queda fuera de ruff (`extend-exclude`): sus nombres no son módulos válidos hasta instanciarla.

`<N>` es el número de la capa (`2`) y `<nombre>` su nombre en minúsculas
(`transform`). Ambos marcadores aparecen en nombres de carpeta y en el
contenido de los archivos. `<descripción>` es un tercer marcador, solo en el
`pyproject.toml`: una frase sobre qué hace la capa.

## Cómo instanciarla

1. Copia la carpeta a `layers/l<N>_<nombre>/`.
2. Reemplaza `<N>` y `<nombre>` en nombres y contenidos, y renombra
   `src/l<N>_<nombre>/`. Comprueba con `grep -rn '<N>\|<nombre>\|<descripción>' layers/l<N>_<nombre>`.
3. Completa `pyproject.toml` (descripción y dependencias), y los `TODO` de
   `smoke.sh` (variables de entorno, modo, rutas de salida y salidas a
   verificar). El humo falla a propósito hasta que lo completes: así el CI no
   da un "humo OK" sin verificar nada. Quita
   `.gitkeep` de `tests/` y `config/` cuando agregues archivos.
4. Agrega `l<N>_<nombre>` a `LAYERS` en `.github/workflows/ci.yml`.
5. Agrega `layers/l<N>_<nombre>/tests` a `testpaths` en el `pyproject.toml` de
   la raíz y la capa a sus `dependencies` y `[tool.uv.sources]`; corre
   `uv lock`.
6. Instancia el módulo `layer` en `infra/stacks/batch/l<N>/main.tf`, con la
   imagen `l<N>_<nombre>:<VERSION>` y sus raíces por variable de entorno
   (ver `infra/stacks/batch/l1/main.tf`).

## Qué no contiene

Código de ninguna capa: solo el esqueleto de arranque.
