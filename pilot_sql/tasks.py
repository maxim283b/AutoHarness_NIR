"""Deterministic SQL benchmark catalogue.

The reference SQL is intentionally kept in the internal records returned by
``load_tasks``.  Agents should use ``public_task``; it never contains it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "benchmark.sqlite"
SCHEMA = """CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, name TEXT, region TEXT, signup_date TEXT);
CREATE TABLE products(product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
CREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT);
CREATE TABLE order_items(order_id INTEGER, product_id INTEGER, quantity INTEGER, unit_price REAL);
CREATE TABLE employees(employee_id INTEGER PRIMARY KEY, name TEXT, department TEXT, manager_id INTEGER, hire_date TEXT, salary REAL);
CREATE TABLE events(event_id INTEGER PRIMARY KEY, customer_id INTEGER, event_date TEXT, event_type TEXT, value REAL);"""

# Twelve broad query families with five genuinely different parameterizations.
# ``template_group`` below splits each family into two groups (3 + 2 variants),
# so near-identical variants can remain together during train/test partitioning.
FAMILIES = [
    ("filter", "Find customers in {region} who signed up on or after {date}.",
     "SELECT customer_id, name FROM customers WHERE region='{region}' AND signup_date>='{date}' ORDER BY customer_id",
     [{"region":"North","date":"2023-01-01"},{"region":"South","date":"2023-01-01"},{"region":"West","date":"2022-01-01"},{"region":"North","date":"2024-01-01"},{"region":"South","date":"2024-01-01"}]),
    ("filter", "Find {category} products costing {op} {price}.",
     "SELECT product_id, name FROM products WHERE category='{category}' AND price {op_sql} {price} ORDER BY product_id",
     [{"category":"Books","op":"less than","op_sql":"<","price":30},{"category":"Office","op":"less than","op_sql":"<","price":20},{"category":"Office","op":"at least","op_sql":">=","price":18},{"category":"Gifts","op":"at most","op_sql":"<=","price":15},{"category":"Books","op":"more than","op_sql":">","price":30}]),
    ("aggregate", "Count customers per region who signed up on or after {date}.",
     "SELECT region, COUNT(*) AS customer_count FROM customers WHERE signup_date>='{date}' GROUP BY region ORDER BY customer_count DESC, region",
     [{"date":"2022-01-01"},{"date":"2023-01-01"},{"date":"2023-06-01"},{"date":"2024-01-01"},{"date":"2024-04-01"}]),
    ("aggregate", "For each category, compute the average price of products costing at least {price}, rounded to two decimals.",
     "SELECT category, ROUND(AVG(price),2) AS avg_price FROM products WHERE price>={price} GROUP BY category ORDER BY category",
     [{"price":0},{"price":10},{"price":15},{"price":20},{"price":40}]),
    ("join", "List {status} orders on or after {date}, with customer name.",
     "SELECT o.order_id,c.name,o.order_date FROM orders o JOIN customers c ON c.customer_id=o.customer_id WHERE o.status='{status}' AND o.order_date>='{date}' ORDER BY o.order_id",
     [{"status":"completed","date":"2024-01-01"},{"status":"completed","date":"2024-03-01"},{"status":"cancelled","date":"2024-01-01"},{"status":"pending","date":"2024-01-01"},{"status":"completed","date":"2024-06-01"}]),
    ("join", "Calculate completed-order revenue per customer and keep totals of at least {amount}.",
     "SELECT c.customer_id,c.name,ROUND(SUM(i.quantity*i.unit_price),2) AS revenue FROM customers c JOIN orders o ON o.customer_id=c.customer_id JOIN order_items i ON i.order_id=o.order_id WHERE o.status='completed' GROUP BY c.customer_id,c.name HAVING SUM(i.quantity*i.unit_price)>={amount} ORDER BY c.customer_id",
     [{"amount":0},{"amount":25},{"amount":40},{"amount":50},{"amount":70}]),
    ("subquery", "Find products priced above {factor} times the overall average price.",
     "SELECT product_id,name,price FROM products WHERE price>{factor}*(SELECT AVG(price) FROM products) ORDER BY product_id",
     [{"factor":0.5},{"factor":0.75},{"factor":1.0},{"factor":1.25},{"factor":1.5}]),
    ("subquery", "Find customers with at least one {status} order on or after {date}.",
     "SELECT customer_id,name FROM customers WHERE customer_id IN (SELECT customer_id FROM orders WHERE status='{status}' AND order_date>='{date}') ORDER BY customer_id",
     [{"status":"completed","date":"2023-01-01"},{"status":"completed","date":"2024-03-01"},{"status":"completed","date":"2024-06-01"},{"status":"cancelled","date":"2024-01-01"},{"status":"pending","date":"2024-01-01"}]),
    ("window", "Rank products by {column} within category, {direction} first.",
     "SELECT product_id,category,{column},RANK() OVER (PARTITION BY category ORDER BY {column} {dir_sql}) AS item_rank FROM products ORDER BY category,item_rank,product_id",
     [{"column":"price","direction":"highest","dir_sql":"DESC"},{"column":"price","direction":"lowest","dir_sql":"ASC"},{"column":"name","direction":"alphabetically earliest","dir_sql":"ASC"},{"column":"name","direction":"alphabetically latest","dir_sql":"DESC"},{"column":"product_id","direction":"largest identifier","dir_sql":"DESC"}]),
    ("window", "Show each employee and the {metric} salary in their department.",
     "SELECT employee_id,department,salary,ROUND({fn}(salary) OVER (PARTITION BY department),2) AS department_value FROM employees ORDER BY employee_id",
     [{"metric":"average","fn":"AVG"},{"metric":"minimum","fn":"MIN"},{"metric":"maximum","fn":"MAX"},{"metric":"total","fn":"SUM"},{"metric":"employee count","fn":"COUNT"}]),
    ("date", "Count orders per month from {start} up to but not including {end}.",
     "SELECT strftime('%Y-%m',order_date) AS month,COUNT(*) AS order_count FROM orders WHERE order_date>='{start}' AND order_date<'{end}' GROUP BY month ORDER BY month",
     [{"start":"2023-01-01","end":"2025-01-01"},{"start":"2024-01-01","end":"2025-01-01"},{"start":"2024-03-01","end":"2025-01-01"},{"start":"2024-01-01","end":"2024-07-01"},{"start":"2024-06-01","end":"2024-08-01"}]),
    ("date", "Find customers who signed up from {start} up to but not including {end}.",
     "SELECT customer_id,name,signup_date FROM customers WHERE signup_date>='{start}' AND signup_date<'{end}' ORDER BY signup_date,customer_id",
     [{"start":"2022-01-01","end":"2023-01-01"},{"start":"2023-01-01","end":"2024-01-01"},{"start":"2024-01-01","end":"2025-01-01"},{"start":"2023-06-01","end":"2024-06-01"},{"start":"2022-01-01","end":"2025-01-01"}]),
]

_ROWS = {
    "customers": [(1,"Ada","North","2023-02-10"),(2,"Boris","South","2023-07-11"),(3,"Chen","North","2024-01-03"),(4,"Dina","West","2023-11-22"),(5,"Eli","South","2024-05-17"),(6,"Fatima","West","2022-12-01")],
    "products": [(1,"SQL Basics","Books",25.0),(2,"Data Atlas","Books",45.0),(3,"Desk Lamp","Office",18.0),(4,"Notebook","Office",8.0),(5,"Query Mug","Gifts",12.0),(6,"Keyboard","Office",55.0)],
    "orders": [(1,1,"2024-01-12","completed"),(2,2,"2024-02-03","cancelled"),(3,1,"2024-03-19","completed"),(4,3,"2024-03-21","completed"),(5,4,"2023-12-30","completed"),(6,5,"2024-05-02","completed"),(7,6,"2024-06-15","pending"),(8,2,"2024-07-07","completed")],
    "order_items": [(1,1,1,25.0),(1,3,2,18.0),(2,2,1,45.0),(3,2,1,45.0),(3,5,2,12.0),(4,4,3,8.0),(5,6,1,55.0),(6,1,2,25.0),(7,3,1,18.0),(8,2,1,45.0)],
    "employees": [(1,"Iris","Engineering",None,"2020-01-10",90000.0),(2,"Jon","Engineering",1,"2022-04-05",70000.0),(3,"Kira","Sales",None,"2019-08-19",82000.0),(4,"Luka","Sales",3,"2024-02-01",60000.0),(5,"Mina","Support",None,"2021-06-30",65000.0)],
    "events": [(1,1,"2024-01-10","view",1.0),(2,1,"2024-02-10","purchase",25.0),(3,2,"2024-02-11","view",1.0),(4,3,"2024-03-20","purchase",24.0),(5,5,"2024-05-01","view",1.0)],
}

def _make_tasks() -> list[dict[str, Any]]:
    tasks = []
    for family_index, (kind, desc, sql, variants) in enumerate(FAMILIES):
        for variant, params in enumerate(variants):
            text = desc.format(**params)
            reference_sql = sql.format(**params)
            task_id = f"sql-{family_index:02d}-{variant:02d}"
            template_group = f"{kind}-{family_index:02d}-{'a' if variant < 3 else 'b'}"
            upper = reference_sql.upper()
            features = {"family": family_index, "type": kind, "joins": upper.count(" JOIN "), "subqueries": int(upper.count("SELECT") > 1), "window": int(" OVER " in upper), "tables": upper.count(" FROM ") + upper.count(" JOIN ")}
            tasks.append({"task_id": task_id, "family": family_index, "template_group": template_group,
                          "variant": variant, "description": text, "schema": SCHEMA,
                          "reference_sql": reference_sql,
                          "allowed_operations": ["SELECT", "WITH", "aggregate", "window_function"] if kind in ("aggregate", "window") else ["SELECT", "WITH"],
                          "features": features, "order_significant": "ORDER BY" in upper})
    return tasks

def load_tasks() -> list[dict[str, Any]]:
    return _make_tasks()

def public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {k: task[k] for k in ("task_id", "description", "schema", "allowed_operations", "features", "template_group")}

def dataset_hash() -> str:
    return hashlib.sha256(json.dumps(load_tasks(), sort_keys=True).encode()).hexdigest()

def rows():
    return _ROWS
