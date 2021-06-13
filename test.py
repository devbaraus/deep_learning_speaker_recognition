# %%
from deep_audio import Audio, Process

# %%

X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection_language(
    'portuguese/processed/psf_24000.json',
    'english/processed/p75_s50/psf_24000.json',
    validation=True, test=False, flat=True)
# %%
print(X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape)
# %%
set_train = set(y_train)
set_validation = set(y_valid)
set_test = set(y_test)
# %%
print(set_train == set_validation, set_train == set_test)
