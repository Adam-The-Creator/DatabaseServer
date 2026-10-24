"""
Server side analysis of the drawings (requirements of the DataAnalysis web application, TEC-02).

An additive module: it only reads the existing data and never changes a table or a drawing document (SC-02, SC-03).
Everything with a calculation is a pure function over plain dicts, so it can be tested without a database.
Results, reports and thumbnails are stored as files (analysis.store).
"""
