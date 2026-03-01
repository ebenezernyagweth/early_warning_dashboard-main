# early_warning_dashboard

This code generates the dashboard where the results of the EW tool are diplayed.

The scripts need to be run in this order:

1. wards_and_counties.py [incorporates the county info to the last results datasets]
2. compute_trends.py [creates the datasets with trend and alert calculation based on the last results]
3. app.py [generates the dashboard (currenlty displayed in render)]


http://0.0.0.0:8080/

