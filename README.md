Look at requirements.txt to see which packages where used. Also note that the Huge R package is also required.

Test folder contains notebooks where I test models and methods

Visualize folder contains notebooks to visualize and infer from data generate from Scripts

To replicate the data preperation and modelling steps of the paper do the following, but note that all data/simulations is provided.

* start by running <code>tidy_esg_data.ipynb</code> which tidies the ESG data, This will generate the <code>esg_refined_no_diff.pkl</code> data, which is already present in  the <code>data/tidy</code> folder
* Next run <code>Scripts/gp_cv_stock.py</code> in a terminal <code> python Scripts\gp_cv_stock.py </code>. This will smooth the ESG data using a GP smoother. Look the output in <code>Visualize/ESG_GP_SMOOTH.ipynb</code>. 
* Run <code> Generate_graphs_case_1.py</code> to get graphs and covariance for each tertile (good, medium, and poor ESG portfolios) for each sector
* Run  <code>paper_1_compare_graphs.py</code> to perform mmd testing. Need to specify the file which contains the graph information which is a output of Generate_graphs_case_1
* Go to  <code>Visualize/Graphs.ipynb</code> to visualze portfolios/graphs and run lasso, SVM, PCA.



