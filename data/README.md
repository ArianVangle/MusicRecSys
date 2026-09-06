# Данные

## Last.fm-360K (основной датасет)

Не хранится в репозитории: 1.6 ГБ.

1. Откройте https://zenodo.org/records/6090214
2. Скачайте `lastfm-dataset-360K.tar.gz`
3. Распакуйте так, чтобы получился путь:
   `data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv`

MD5: `be672526eb7c69495c27ad27803148f1`

Формат (табуляция, без заголовков):
```
user-mboxsha1 \t musicbrainz-artist-id \t artist-name \t plays
```

Данные предоставлены Last.fm для некоммерческого использования,
собраны Oscar Celma.

## MovieLens-100k (кросс-доменная проверка)

Файлы `u.data` и `u.item` лежат в репозитории (2 МБ), скачивать не нужно.
Нужны для воспроизведения таблицы сравнения доменов и временного сплита.

Источник: https://files.grouplens.org/datasets/movielens/ml-100k.zip
GroupLens Research, University of Minnesota.
