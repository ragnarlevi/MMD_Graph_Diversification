Look at requirements.txt to see which packages where used. Also note that the Huge R package is also required.

Test folder contains notebooks where I test models and methods

Visualize folder contains notebooks to visualize and infer from data generate from Scripts

To replicate the data preperation and modelling steps of the paper do the following, but note that all data/simulations is provided.

* start by running <code>tidy_esg_data.ipynb</code> which tidies the ESG data, This will generate the <code>esg_refined_no_diff.pkl</code> data, which is already present in  the <code>data/tidy</code> folder
* Next run <code>Scripts/gp_cv_stock.py</code> in a terminal <code> python Scripts\gp_cv_stock.py </code>. This will smooth the ESG data using a GP smoother. Look the output in <code>Visualize/ESG_GP_SMOOTH.ipynb</code>. 
* Run <code> Generate_graphs_case_1.py</code> to get graphs and covariance for each tertile (good, medium, and poor ESG portfolios) for each sector. The output can be found in <code>data\Graphs </code> for all sectors but we had to leave out the global study due to size limitations (but we did include the striped version which contains less data) 
* Run  <code>paper_1_compare_graphs.py</code> to perform mmd testing an generates a time series of MMD p-values. Need to specify the file which contains the graph information which is a output of Generate_graphs_case_1. Note that we used the data with the appendix <code> striped </code> , which can be found in <code>data\Graphs </code>, to generate the MMD p-value time series. 
* Go to <code>Visualize/Graphs.ipynb</code> to visualze portfolios/graphs and run lasso, SVM, PCA. Note that due to size limitations we could not upload all. Note that because we could not include the full graph data for the global study, some of the cells, namely the ones plotting graph statitics, won't work for the global study



