# Wakeel LLM

## Summary
Provide a brief overview of the project, its purpose, and key objectives.

## Data Sources

https://www.pakistanlawsite.com/Login/Check
https://www.peshawarhighcourt.gov.pk/PHCCMS/reportedJudgments.php?action=search

## Features
4/13/2025
 1. TAB 1: LEGAL CONSULTING: This tab is for both general users and lawyers. 
 Input: Text, Attach input as pdf. 
 Method: Fine-tuned Model
 Output: Summary/Explanation of the input document in concise/easy manner.
 2. TAB 2: DRAFTING:This tab is for lawyers, allowing them to draft e.g. inhertance Will Deed.
 Input: Text, Attach input as pdf. 
 Method: Prompting (Possibly RAG)
 Output: Draft for particular case. (Pilot draft: Will Deed)
 3. TAB 3: CITATIONS: Given the input, give the relevant cases.
 Input: Text
 Method: RAG
 Output: Top 3 relevant citations with content summary and sources.
