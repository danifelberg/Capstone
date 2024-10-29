Note: Use Markdown Cheat Sheet if you need more functionality
https://www.markdownguide.org/cheat-sheet/
### Date: sep 17 2024 

**- Action Items:**
* [X] Add more scrapers (or enhance current one) to obtain more headlines
* [X] Add separate code for classes/functions
* [X] Add readme file with progress report and action items
* [ ] Continue developing AR model
    * [ ] Evaluate AIC/BIC
    * [ ] Add exogenous input
---
### Date: sep 24 2024 
    - Cloned repository to local desktop for organizing
    - Modified class and utilities python scripts
        - Added stationarity functions and initial scraping class
    - Made scraper loopable to include other results pages
    

**- Action Items:**
* [X] Rename and reorganize repo from "Sample Capstone"
* [kind of] Expand scraper to include more headlines
* [ ] Develop AR/ARX model
* [ ] Continue organizing script into classes, utils, etc.
---
### Date: oct 1 2024 
- Repo organization
- AR/ARX model evaluation
- News API
- Meet again this week? -> No


**- Action Items:**
* [X] Fix ARX and make sure base AR works
* [kind of] News API code (Tim will send) --> extract headline (expanded on FT scraper instead)
* [X] Use embedding vector (sequence-to-sequence) --> sentence bert
* [O] Model class separately, EDA class, metric class, News API class
* [O] Main file combining code (should not be more than just a few lines)
---
### Date: oct 8 2024
- Repo organization
- AWS setup?
- Seq2Seq (maybe) --> didn't use sentence bert, used pytorch instead


- Action Items:
* [X] ORGANIZE REPO!!!! --> no more giant script!!!
   * [X] Clean up code --> models should be separate, EDAs should be separate, etc. (everything should be 1 line)
* [O] Create classes for EDA, metrics, News API --> not classes, but separate utils files
* [O] Fix graph indices (train, test, prediction should be overlapping)
* [O] **Test out AR model code (use sample stationary synthetic data)**
* [O] Grid search for AR (ACF/PACF is not enough)

---
### October 15: 
- Repo organization (what to improve)
- How to fix NewsAPI scraping (currently returns error code 400)


- Action Items:
* [X] Work on the full pipeline (ignore ARX/headlines) --> figure out ideal model for JUST stock time series
* [X] Continue organizing repository
* [X] Automate grid search so it adjusts depending on how much/often stock data is fetched
---
### Date: October 22
- Need to improve AR(2)?
   - Why is it outperforming model with lower RMSE?
- Next steps for ARX
- Improve organization?


- Action Items:
* [X] For the same time period as AR model, check to see if ARX, SARIMAX, and LSTM perform better --> LSTM seems promising, SARIMAX less so

### Date: October 29

- Debugging SARIMAX code
- Next steps (which models to use)

- Action Items:
[ ] Finish debugging SARIMAX (can include in final paper if results are promising)
[ ] Finalize LSTM without headlines, and create new LSTM that includes headlines as an embedding (add dimensionality, values change depending on if date has a headline attached to it)
[ ] Once LSTM is finalized, work on GRU (with and without headlines)
[ ] Lastly, work on transformers (also with and without headlines)!
