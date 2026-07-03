# Reproducir la evaluación del clasificador (e / π / μ) sobre test-beam

Cómo replicar los resultados a partir del **modelo ya entrenado** y de los datos
de test-beam. Se asume que ya tienes el hardware (GPU NVIDIA con CUDA — el forward
de GATr usa `xformers` y **solo corre en CUDA**) y el entorno de ejecución
(contenedor Apptainer `gatr_v9.sif` + `extlib/`, o un entorno equivalente Python 3.8
con torch/gatr/xformers/torch-geometric/h5py/numpy/pyyaml/matplotlib).

---

## 1. Código (del repo)

Solo la rama de evaluación:

- `src/evaluate_classifier.py` — entry point. **El fix thr1↔thr2 del test-beam ya
  está dentro** (las etiquetas de threshold 1 y 2 vienen intercambiadas por error en
  el fichero de TB; el script lo corrige al construir el one-hot).
- `src/models/gatr_autoencoder.py`, `src/models/classification_head.py`,
  `src/models/gatr_module.py`, `src/models/attention_pooling.py` (los dos últimos son
  imports transitivos del autoencoder)
- `src/utils/batch_utils.py`, `src/utils/clf_data.py`, `src/utils/datasets.py`
  (`datasets.py` es import transitivo de `clf_data.py`)
- `src/__init__.py`, `src/utils/__init__.py` (necesarios para que resuelvan los
  imports de paquete)

## 2. Artefactos entrenados (por energía)

| Fichero | Qué es | Tamaño |
|---|---|---|
| `checkpoints_clf_finetune_<E>GeV_p1e5/finetune_best.pt` | AE fine-tuned + cabeza (bundle) | ~28 MB |
| `config/model_cfg_clf_finetune.yml` | arquitectura (debe **coincidir** con el ckpt) | <5 KB |
| `config/clf_combined_train_noisy_p1e5_<E>GeV_stats.yml` | stats z-norm (train y eval **comparten** el mismo fichero) | <1 KB |

No hace falta el backbone VAE ni los datos de sim/ruido: todo está fundido en
`finetune_best.pt`. `<E>` ∈ {20, 50, 80}.

## 3. Datos de test-beam

- `data_jorge_flat_subset9.h5` (~155 MB) — subset del `data_jorge_flat.h5` original
  (12 GB) con las **9 energías** que se usan (15/20/25/40/50/60/70/80/90 GeV), 20 000
  eventos por energía, mismo esquema (`energy, beam_type, offsets, i, j, k, thr,
  x, y, z`) y sin filtrar (el fix thr1↔thr2 lo aplica el propio script).
- Apunta `--testbeam_path` a este fichero.
- Con los flags por defecto (`--max_events_tb 50000`) el eval hace un subsampleo
  aleatorio determinista (`seed=42`) sobre el subset → plots estadísticamente
  equivalentes a los de referencia. Para usar todos los eventos del subset:
  `--max_events_tb 0`.

## 4. Comando

Cada modelo se evalúa sobre el triplete de energías de TB alrededor de su energía
de entrenamiento.

**20 GeV** (test-beam 15/20/25):
```bash
python src/evaluate_classifier.py \
  --finetune_ckpt checkpoints_clf_finetune_20GeV_p1e5/finetune_best.pt \
  --cfg config/model_cfg_clf_finetune.yml \
  --stats_yaml config/clf_combined_train_noisy_p1e5_stats.yml \
  --testbeam_path <ruta>/data_jorge_flat.h5 \
  --use_scalar --use_one_hot --z_norm --latent_source aggregate \
  --tb_energies 15 20 25 --min_hits_tb 20 --max_events_tb 50000 \
  -o eval_20GeV --device cuda:0
```

**50 GeV** (test-beam 40/50/60):
```bash
python src/evaluate_classifier.py \
  --finetune_ckpt checkpoints_clf_finetune_50GeV_p1e5/finetune_best.pt \
  --cfg config/model_cfg_clf_finetune.yml \
  --stats_yaml config/clf_combined_train_noisy_p1e5_50GeV_stats.yml \
  --testbeam_path <ruta>/data_jorge_flat.h5 \
  --use_scalar --use_one_hot --z_norm --latent_source aggregate \
  --tb_energies 40 50 60 --min_hits_tb 20 --max_events_tb 50000 \
  -o eval_50GeV --device cuda:0
```

**80 GeV** (test-beam 70/80/90):
```bash
python src/evaluate_classifier.py \
  --finetune_ckpt checkpoints_clf_finetune_80GeV_p1e5/finetune_best.pt \
  --cfg config/model_cfg_clf_finetune.yml \
  --stats_yaml config/clf_combined_train_noisy_p1e5_80GeV_stats.yml \
  --testbeam_path <ruta>/data_jorge_flat.h5 \
  --use_scalar --use_one_hot --z_norm --latent_source aggregate \
  --tb_energies 70 80 90 --min_hits_tb 20 --max_events_tb 50000 \
  -o eval_80GeV --device cuda:0
```

Salida en `-o`: los plots `tb_aggregate_*.png`, `tb_summary.txt` y (si se añade
`--save_predictions <out>/tb_predictions_raw.npz`) las predicciones crudas.

## 5. Puntos críticos (fallos típicos)

1. Los flags `--use_scalar --use_one_hot --z_norm` y `--latent_source aggregate`
   deben ir **tal cual**: definen las features de entrada. Cambiarlos rompe el
   z-norm / one-hot respecto al entrenamiento.
2. `--cfg` y `--stats_yaml` tienen que ser los del **mismo** modelo/energía.
   Mezclarlos desnormaliza la entrada **sin lanzar error** → resultados malos y
   difíciles de diagnosticar.
3. Sin GPU CUDA no arranca (GATr/xformers).

> Contexto de por qué el pipeline es así (fix de threshold + nivel de ruido
> `p_noise=1e-5`): ver `DOMAIN_ADAPTATION.md`.
