# mphys-galaxy
 Using group-equivariant deep-learning to determine the chirality of spiral galaxies


Note: Models are stored and logged as follows:

Logs:
[Model name]_[dataset_name]_[custom]
	version_[run]_[test,train or predict]
		metrics.csv (csvlogger)
		hparams.yml

Metrics:
[Model name]_[dataset_name]_[custom]
	version_[run]
		train_metrics.csv
		test_metrics.csv
		model.pt
		val_matrix.png
		test_matrix.png
		test_metrics.csv