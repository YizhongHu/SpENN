# data_structures instructions

Feature storage should use canonical `Partition` keys internally, while
accepting tuple/list/string/int partition specs at API boundaries. Keep shape,
device, dtype, batch, and electron-count validation centralized.
