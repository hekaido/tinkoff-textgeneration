# tinkoff-textgeneration
Мое решение задачи Тинькофф по генерации текстов

Обучал модель на тексте И.А.Гончарова "Обломов"

Шаги обучения:
1) Считыание файлов, токенизация, отсев предложений по длине
2) Создание словаря
3) Обучение word2vec
4) Разделение предложения на части для обучения и генерации, усреденение их
5) Обучение линейного слоя
6) Сохранение модели