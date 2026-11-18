# SQL benchmark: smoke report

Дата: 2026-10-08  
Seed: `20261008`  
Harness: `direct`  
Задач: 5  

Результат: `5/5 success`. Это техническая самопроверка verifier: callback возвращал эталонный SQL. Это **не** измерение качества LM Studio или `direct` harness. Полная матрица harness, обучение выбора и held-out задачи не запускались.

Проверки verifier-а: корректный эталонный запрос проходит; неправильный результат получает `wrong_result`; `INSERT`/`UPDATE`/`DELETE`/`DROP` и несколько statements блокируются; порядок строк проверяется там, где он значим; долгий recursive query получает структурированный результат без раскрытия эталона.

`smoke_test` не перезаписывает уже существующий `pilot_sql/data/smoke_results.jsonl`: повторный запуск завершился защитным сообщением. Это ожидаемое поведение для воспроизводимого журнала.

Проверено командой:

```text
python -m pytest pilot_sql/tests -q
5 passed in 0.07s
```
