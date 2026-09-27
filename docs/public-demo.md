# Публичная демонстрация

[Открыть AIMedicine](http://51.250.20.30:8502)

Интерфейс показывает сохранённые истории трёх пациентов, эталонный offline-прогон и тесты внесённых ошибок. Доступны карточка и динамика, версии планов, изменения, аудит и тексты источников.

В списке «Папка запуска»:

- `reports/live` — завершённые случаи с Mistral.
- `reports/evaluation_live` — независимый аудит ошибки дозы и нарушений процедурных политик.
- `reports/evaluation` — тестовые варианты для проверки правил.
- `reports/offline` — воспроизводимые эталонные случаи.

Публичный интерфейс работает в режиме просмотра. Локальный на `127.0.0.1:8501` также поддерживает новые запуски.

## Служба

Пользовательская служба systemd запускается автоматически и перезапускается при сбое. Linger позволяет ей работать независимо от SSH-соединения.

```bash
cd ~/Desktop/MAGA/AIMedicine
systemctl --user link "$PWD/deploy/aimedicine-public.service"
systemctl --user daemon-reload
systemctl --user enable --now aimedicine-public.service
sudo ufw allow 8502/tcp comment 'AIMedicine public demo'
```

## Управление

```bash
systemctl --user status aimedicine-public.service
journalctl --user -u aimedicine-public.service -n 50 --no-pager
systemctl --user restart aimedicine-public.service
systemctl --user stop aimedicine-public.service
```

Проверка доступности: `http://51.250.20.30:8502/_stcore/health`. Результаты функциональных проверок приведены в [отчёте](validation.md).
