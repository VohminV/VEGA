# VEGA

**VEGA** (Vision Engine — просто короткое и звучное название). Собственный
детектор объектов на PyTorch. Без ultralytics, mmdetection, detectron2
и других готовых детекторных библиотек — своя сеть и свой pipeline.

Читает ваш существующий YOLO-датасет как есть (`images/`, `labels/`, `data.yaml`),
ничего не конвертирует и не создаёт тайлового датасета.

## Установка

```
pip install -r requirements.txt
```

## Как указать путь к датасету

Отредактируйте `configs/default.yaml`, секцию `dataset`:

```yaml
dataset:
  root: "/path/to/my_dataset"   # корень вашего датасета
  images_dir: "images"
  labels_dir: "labels"
  data_yaml: "data.yaml"
```

Датасет может быть в двух раскладках:

1. `root/images/*.jpg` + `root/labels/*.txt` + `root/data.yaml` без train/val —
   тогда все изображения используются как обучение, а 10% случайно уходит в валидацию.
2. Стандартный YOLO-разрез: в `data.yaml` указаны `train:` и `val:` (списки или строки
   путей, абсолютные или относительно `root`). Лейблы ищутся по соглашению:
   `images/...` → `labels/...` с тем же относительным путём и расширением `.txt`.

Классы задаются в `configs/default.yaml` (`dataset.classes`) или читаются из `names:`
в `data.yaml`. Число классов берётся из `model.num_classes`.

## Как запустить обучение

```
python train.py --config configs/default.yaml
```

Логи пишутся в `runs/train/log.txt`, веса в `runs/train/last.pt` и `runs/train/best.pt`
(лучший по mAP@0.5). После каждой эпохи — быстрая валидация.

## Как запустить валидацию

```
python val.py --config configs/default.yaml --weights runs/train/best.pt
```

Результаты: `runs/val/metrics.json`, `runs/val/metrics.txt`, а также
`runs/val/missed_small.txt` и примеры пропущенных мелких объектов в `runs/val/missed_examples/`.

## Как запустить предсказание

```
python predict.py --config configs/default.yaml --weights runs/train/best.pt --source /path/to/image.jpg
python predict.py --config configs/default.yaml --weights runs/train/best.pt --source /path/to/folder
python predict.py --config configs/default.yaml --weights runs/train/best.pt --source /path/to/video.mp4 --source_type video
```

Для изображений/папки результат: `runs/predict/images/*.jpg` (аннотации) и
`runs/predict/labels/*.txt` (предсказания в YOLO-формате `class_id cx cy w h [conf]`).
Для видео: `runs/predict/<имя>_annotated.avi` и `runs/predict/video_labels/*.txt`.

## Как экспортировать в ONNX

```
python export_to_onnx.py --weights runs/train/best.pt --config configs/default.yaml
python export_to_onnx.py --weights runs/train/best.pt --config configs/default.yaml --batch 4
```

Экспортируется сама сеть: три выхода `out_s4/out_s8/out_s16` по одному на уровень пирамиды,
каждый `(B, 1+4+C, H, W)` (objectness logit, dx/dy/dw/dh, class logits).
Декодирование и NMS выполняются снаружи.

## Настройки, важные для мелких удалённых объектов

- `val.use_tiled_inference` / `predict.use_tiled_inference: true` — разрезать вход на
  тайлы с перекрытием (`tile_size`, `overlap_ratio`). Сильно повышает находимость мелких
  объектов без потери детализации.
- `model.input_size: 1024` — можно поднять до тренировочного разрешения вашего датасета.
  Выше — лучше для мелочи, но тратит больше VRAM.
- `model.strides: [4, 8, 16]` — уровень с stride=4 отвечает за мелкие объекты, не удаляйте его.
- `val.conf_threshold: 0.15` — ниже порог — больше находок (но и больше ложных). 
- `augmentation.scale_crop: 0.5` + `scale_min/max` — случайный crop+масштаб на
  обучении: разнообразие размеров целей, помогает мелким объектам.
- Маленькие боксы из датасета **никогда не удаляются**: цель назначается ячейке центра
  плюс соседним ячейкам, если объект меньше шага сетки.
- `small_object.small_min_side_px` и `small_object.small_area_fraction` определяют, что
  считается «мелким объектом» для отдельных метрик small_precision/small_recall.

## Что делать, если мало VRAM

- Уменьшите `model.input_size` (например, до 640).
- Уменьшите `train.batch_size` до 1–2.
- Выберите лёгкий backbone: `model.backbone: custom_small` уже самый лёгкий.
- Поставьте `train.amp: true` (по умолчанию) — AMP экономит память и ускоряет.
- Поставьте `train.workers` в 0, если данные на медленном диске.

## Что делать, если мелкие объекты теряются

1. Включите тиловый инференс на валидации и предикте (`use_tiled_inference: true`).
2. Снизьте `val.conf_threshold` и `predict.conf_threshold`.
3. Поднимите `model.input_size`.
4. Проверьте small_recall в val — если он заметно ниже общего recall, дело в мелких.
5. Увеличьте `model.fpn_channels` / `model.head_channels`, если позволяет VRAM.

## Структура

```
VEGA-detector/
  README.md
  requirements.txt
  train.py            # обучение + быстрая валидация
  val.py              # полная валидация с метриками по мелким объектам
  predict.py          # инференс: изображение / папка / видео (+ tiled)
  export_to_onnx.py   # экспорт модели в ONNX
  configs/default.yaml
  src/
    config.py         # чтение конфига
    dataset.py        # YOLO-датасет, фильтрация битых строк, аугментации
    model.py          # своя сеть: backbone + FPN + голова (strides 4/8/16)
    losses.py         # focal_Bce objectness, GIoU box, BCE class + target assignment
    metrics.py        # precision/recall/mAP@0.5 (per-image AP) + small-метрики
    boxes.py          # разбор лейблов, IoU, NMS (class-aware), letterbox
    tiled_inference.py# инференс с тайлами (batch) и итоговым NMS
  tests/              # pytest: equivalence _assign_level, аугментации, NMS, LR, decode
```

## Тесты

```
python -m pytest tests/ -q
```

Покрывают: эквивалентность векторизованного назначения целей, аугментации
(scale/crop и схлопывание бокса после clip), NMS, LR-расписание, декодирование регхидов.