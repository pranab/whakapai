# whakapai
This project is about Big Data ETL processing on spark. I have various Map reduce jobs in different 
projects, essentially performing ETL processing. My plan is to consolidate them under one 
project and have them running on Spark. I will be adding new ETL spark jobs also as I go along.

This project is meant for large data volume ETL processing. Unlike many other ETL tools, there
won't be any visual tool for manual processing of data. It will use traditional ETL processing and
machine learning where necessary

Here are the planned features for now.


## Data validation
1. Field level validation with regular expression and custom logic
2. Inter field level or record level validation with custom logic
3. Isolation of invalid records and merge back after correction

## Outlier detection
1. Field level
2. Record level
3. Various statistical and proximity based algorithms

## Missing data processing
1. Isolating records with missing fields
1. Replace missing fields with imputation

## Deduplication and record linkage
1. Normalizing structured text field according to country
2. Various free form and structured text matching algorithms

## Data format
1. Flat record
2. JSON
3. XML