Look at requirements.txt to see which packages where used. Also note that some R packages are required such as spectralGraphTopology, fingraph, fitHeavyTail


Test folder containts notebooks where I test models and methods

Visualize folder contains notebooks to visualize and infer from data generate from Scripts

To replicate the data preperation and modelling steps of the paper

* start by running tidy_esg_data.ipynb which tidies the ESG data and creates a sector ESG index.
* Next run Scripts/gp_cv_stock.py and Scripts/gp_cv_sector.py using a terminal. This will smooth the ESG data using a GP smoother. Look at Visualize/ESG_GP_SMOOTH.ipynb and run the code to generate a data frame of daily ESG scores for stocks and sectors
* Run Generate_graphs_case_1 to get graphs and covariance for each tertile for each sector
* Run paper_1_compare_graphs.py to perform mmd testing. Need to specify the file which contains the graph information which is a output of Generate_graphs_case_1





