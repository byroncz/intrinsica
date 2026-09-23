# scaffold/layer_template/

Plantilla para crear una capa nueva. Definido en el
[TRD maestro §8.1](../../docs/TRD/plataforma_directional_change.md#81-estrategia-de-repositorio-monorepo-políglota)
y §8.6 (playbook).

## Regla (ADR-07)

**Una imagen por capa, modo por parámetro.** Los modos (por ejemplo
`backfill`, `daily`, `monthly-close`) se eligen al ejecutar, no con imágenes
distintas.

## Árbol esperado de una capa

```
l<N>_<nombre>/
├── Dockerfile        # multi-stage, base slim/distroless
├── .dockerignore
├── pyproject.toml
├── VERSION           # versión semántica de la capa
├── src/
├── tests/
└── config/           # YAML versionados
```

## Qué no contiene

Código. La plantilla se extraerá de la primera capa real (E2).
