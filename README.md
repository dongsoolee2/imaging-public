# imaging

Optical calcium imaging data management pipeline in Python

Dongsoo Lee

Baccus Lab, Stanford University



## Data Management Pipeline

Extracted calcium responses, `*.mat`, are transformed (cleaned and sanitized) \
and combined. The final output `*.h5` files are loaded into linear models and \
Deep Learning models (`im-torch` repo).


## Linear Models

Linear models capture linear weights between input and output. By using reverse \
correlation methods, linear receptive field can be computed.
