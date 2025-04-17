# Wakeel LLM

## Summary
Provide a brief overview of the project, its purpose, and key objectives.

## Data Sources

https://www.pakistanlawsite.com/Login/Check
https://www.peshawarhighcourt.gov.pk/PHCCMS/reportedJudgments.php?action=search

## Features
4/13/2025
 1. TAB 1: LEGAL CONSULTING: This tab is for both general users and lawyers. It has two buttons: RAG & FT
 Input: Text, Attach input as pdf.(For demo: Text input only)  
 Method: Fine-tuned Model, RAG-enhanced model
 Output: Summary/Explanation of the input document in concise/easy manner.
 2. TAB 2: DRAFTING:This tab is for lawyers, allowing them to draft petitions e.g. Khulaa Petition.
 Input: Text
 Method: Agentic AI, Prompting
 Output: Draft in text file for a particular case. (Pilot draft: Khula Petition)
 3. TAB 3: CITATIONS: Given the input, give the relevant cases.
 Input: Text
 Method: RAG
 Output: Top 3 relevant citations with content summary and sources.
