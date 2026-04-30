# Debt

Here, I list shortcuts/decisions made during the development phase of the algorithm. These need to be accounted for later on so that the app has a working capture flow.

## Background Image

Taken as a median of a shots frames or over the "session's" shots. In reality we should have a calibration phase that builds the background.

## Region of interest

A. This should be tied to the initial location of the ball instead of hardcoding it OR B. we choose a comfortable region and force the user to set up the camera so that it respects the region.

Right now we have chosen to go with B.
