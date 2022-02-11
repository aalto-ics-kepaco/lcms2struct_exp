####
#
# The MIT License (MIT)
#
# Copyright 2021 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####

# Task: Query the Classyfire molecule classes for the MassBank entries and 
#       insert them in the database

require(classyfireR)
require(RSQLite)

# Read command line arguments
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 1) {
  stop("Usage: R insert_classyfire_classes DB_FILE")
}
db_fn <- args[1]

# Check that the DB-file exists
stopifnot(file.exists(db_fn))

# Open database and get the InChIKeys of all MassBank entries to be queried with
# ClassyfireR
con <- dbConnect(RSQLite::SQLite(), db_fn)
tryCatch({
  res <- dbSendQuery(
    con,
    "
    SELECT DISTINCT cid, inchikey FROM scored_spectra_meta
      INNER JOIN molecules m ON m.cid = scored_spectra_meta.molecule
    "
    )
  mols <- dbFetch(res)
  dbClearResult(res)
}, finally = {
  dbDisconnect(con)
})

# Query the molecule classes
mols$kingdom <- NA
mols$superclass <- NA
mols$class <- NA
mols$level_5 <- NA
mols$level_6 <- NA
mols$level_7 <- NA
mols$level_8 <- NA

for (idx in seq_len(nrow(mols))) {
  ikey <- mols[idx, "inchikey"]

  res <- get_classification(ikey)
  if (is.null(res)) { print("bla") }
  
  tmp <- unclass(classification(res))$Classification
  n_cls <- min(length(tmp), ncol(mols) - 2)
  mols[idx, 2 + (1:n_cls)] <- tmp[1:n_cls]
}  

# Open DB to write molecules classes
con <- dbConnect(RSQLite::SQLite(), db_fn)
tryCatch({
  # Enable foreign key support
  dbClearResult(dbSendQuery(con, "PRAGMA foreign_keys = ON"))
  
  # Insert new table
  tryCatch({
    dbBegin(con)
    dbClearResult(dbSendStatement(
      con, 
      "
      CREATE TABLE IF NOT EXISTS classyfire_classes (
        molecule   INTEGER PRIMARY KEY NOT NULL, 
        kingdom    VARCHAR,
        superclass VARCHAR,
        class      VARCHAR,
        level_5    VARCHAR,
        level_6    VARCHAR,
        level_7    VARCHAR,
        level_8    VARCHAR,
        FOREIGN KEY (molecule) references molecules(cid)
      )
      "
    ))
    dbCommit(con)
  }, error = function(error_condition) {
    print("FAILED to insert new table.")
    dbRollback(con)
    traceback()
    stop(error_condition)
  })
  
  # Insert Classyfire classes
  tryCatch({
    dbBegin(con)
    res <- dbSendStatement(
      con,
      "
      INSERT OR REPLACE INTO classyfire_classes VALUES ($cid, $kingdom, $superclass, $class, $level_5, $level_6, $level_7, $level_8)
      "
    )
    dbBind(res, as.list(subset(mols, select=-inchikey)))
    if (dbGetRowsAffected(res) != nrow(mols)) {
      stop(
        "
        Insert molecule classes: Number of effected rows is not equal the number
        of molecules.
        "
      )
    }
    dbClearResult(res)
    
    dbCommit(con)
    print("Classyfire classes inserted.")
  }, error = function(error_condition) {
    print("FAILED to insert classyfire classes.")
    dbRollback(con)
    traceback()
    stop(error_condition)
  })
  
}, finally = {
  dbDisconnect(con)
})
