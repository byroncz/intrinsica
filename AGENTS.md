# intrinsica

Instrucciones para cualquier agente que trabaje en este repo. `CLAUDE.md`
importa este archivo; Codex lo lee directamente. Es la única fuente.

## Qué es este proyecto

intrinsica es un motor de análisis Directional Change (DC) para mercados
financieros: en vez de agrupar precios en barras de tiempo fijo, detecta
puntos de inflexión en la serie y construye indicadores a partir de esos
eventos. La arquitectura de datos sigue el patrón Medallion (Bronze, Silver,
Gold). El proyecto se rediseña desde cero a partir de 2026-09; lo que existe
antes de esa fecha vive en `legacy/v0-local` como referencia, no como diseño
vigente.

## Cómo se trabaja aquí

- Código del proyecto en Notion: el valor `project` de `.devkit/devkit.toml`.
  Cada tarea es una card con Clave `<CÓDIGO>-<n>`. Las skills en
  `.claude/skills/` definen cada paso.
- Una card activa por sesión. Rama `<tipo>/<CÓDIGO>-<n>-slug` desde `main`
  (`feat/`, `fix/` o `chore/` según el Tipo de la card), PR a `main` con
  auto-merge. Nunca push directo a `main`. Nunca force push.
- Sin una card activa en `En progreso` sobre la rama actual, la sesión no
  edita archivos de código. Puede crear cards (`task-create`), comentar,
  revisar (`pr-review`) y escribir Documentación. Todo cambio de código
  entra por una card y su rama; el ciclo automático lo revisa y lo mergea.
  Única excepción: autorización expresa del humano en la conversación.
- Commits con Conventional Commits y la Clave como ámbito:
  `feat(<CÓDIGO>-42): agregar carga incremental`.
- `sandbox.local/` es un espacio de pruebas respaldado en Dropbox y fuera de
  git. Cualquier otro directorio `*.local` no se respalda y muere en el
  rebuild: no guardes ahí nada que importe.
- Python lo gestiona `uv`. Versión en `.devkit/devkit.toml` (clave `python`).
  Dependencias con `uv add`, entorno con `uv sync`, ejecutar con `uv run`.
- Sin `sudo`. Si falta un paquete de sistema, se declara en
  `.devkit/devkit.toml` (`apt`) y se reconstruye la imagen con
  `devkit rebuild`.
- Si una conexión falla con "connection refused", el dominio no está en la
  lista blanca del proxy. Ejecuta `devkit-net-denied`, añádelo a `domains`
  en `.devkit/devkit.toml` y aplica con `devkit recreate`.

## Notion y skills

- Notion es el centro de tareas. Identificadores de las bases (Proyectos,
  Tareas, Documentación) en `.claude/devkit-notion.json`. Accede con el
  plugin oficial de Notion; si no responde, avisa al humano y no improvises.
  Para encontrar una card por Clave, filtra por `ID` y `Proyecto`, no por
  la fórmula `Clave`: el MCP no la devuelve. Detalle en `.claude/skills/README.md`.
- Cada skill en `.claude/skills/` es un paso del flujo. Las principales:
  `/project-status` dice en qué va el proyecto, `/task-start` toma una card
  libre, `/task-submit` entrega el trabajo y abre el PR, `/task-document`
  escribe la entrada de Documentación al aprobar. Cerrar y bloquear no son
  skills sino scripts bash: `task-close.sh` tras el merge y `task-block.sh
  <Clave> "<motivo>"` cuando necesitas al humano, ambos en
  `${DEVKIT_SCRIPTS_DIR:-/opt/devkit/scripts}`. Lee `.claude/skills/README.md`
  para el resto.
- Una skill se edita en el repo del template (DEVKIT), por su ruta real
  `devkit/agents/skills/<skill>/SKILL.md`, nunca por `.claude/skills/`: ese
  directorio es un enlace al template y Claude Code no acepta escrituras bajo
  `.claude/` sin confirmación del humano, que en headless nadie da.
- El humano decide dos cosas: mover cards de Por refinar a Backlog o Lista
  (y Épicas de Backlog a Lista) y aprobar el PR. Todo lo demás lo haces tú,
  sin preguntar, siguiendo las skills.
- En modo headless (`claude -p`) no hay quien responda: una pregunta al
  humano equivale a bloquear la card. Nunca termines con una pregunta
  abierta. Si falta algo, ejecuta `task-block.sh` con el motivo "Qué intenté:
  ... Qué necesito: ..." y la petición concreta (o comenta en
  la card, si no hay card que bloquear) y termina. Toda ejecución headless
  cierra en un estado observable de la card, nunca a la espera.

## Guía de redacción

Aplica a todo texto que escribas: descripciones, comentarios, respuestas,
PRs y entradas de Documentación. Sin excepción.

- Español latino neutro.
- Conciso, simple, autoexplicativo y pedagógico. Escribe para un ingeniero
  de datos de primer año que llega hoy al proyecto.
- Respeta los tecnicismos y las definiciones.
- Cuando expliques una decisión, toma posición crítica y técnica, con
  evidencia contrastada.
- Profundidad proporcional al artefacto:
  - Comentario de avance en una card: dos a cuatro líneas. Qué se hizo y qué
    sigue.
  - Descripción de PR: qué cambia, cómo probarlo, enlace a la card.
  - Entrada de Documentación: completa. Qué cambió, por qué, cómo probarlo,
    cambios requeridos, enlaces. El porqué de cada decisión se escribe aquí
    una sola vez; los demás textos enlazan.

## Reglas del proyecto

<!-- Todo lo de arriba es del template: `template-update` lo reemplaza en cada versión nueva. Esta sección y lo que sigue es del proyecto y se conserva tal cual. -->

- `legacy/v0-local` es solo lectura y referencia. No se importa ni se porta
  código desde ahí a menos que una card lo pida explícitamente.
- No se versionan binarios ni wheels en el repo.
- Eficiencia de memoria ante todo: todo dato alojado en RAM se libera en cuanto
  fue aprovechado; nunca conviven dos representaciones del mismo dato; el pico
  de memoria de una unidad es O(lote), no O(unidad). Aplica a todas las capas.
  Decisión y porqué: [Decisión: eficiencia de memoria ante todo](https://app.notion.com/p/3e727957d23d811887eaf14c886b9a0c)
  (2026-09-26).
- El stack (lenguajes, librerías, motor de cómputo, almacenamiento) lo fija
  la entrada de Documentación tipo decisión correspondiente. Por ahora no
  existe esa entrada; cuando se cree, enlazarla aquí.
- Los agentes solo abren PRs. No disparan `terraform.yml`, `run-job.yml` ni
  ningún `workflow_dispatch`. `apply`, `destroy` y la ejecución de jobs los
  dispara y aprueba el humano en el environment `gcp`; el stack `data` lo
  aplica solo el humano desde Cloud Shell. Ningún agente instala `gcloud` ni
  guarda credenciales de GCP: la única identidad es WIF, dentro de Actions.
  Leer runs y logs con `gh run list` y `gh run view --log` sí está permitido.
  Decisión y porqué: [Decisión: despliegue de stacks de capa por GitHub
  Actions con WIF](https://app.notion.com/p/3e527957d23d810e9401d9d941d17f83)
  (2026-09-24).

