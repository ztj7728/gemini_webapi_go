### Logging Configuration

This package uses [loguru](https://loguru.readthedocs.io/en/stable/) for logging, and exposes a function `set_log_level` to control log level. You can set log level to one of the following values: `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`. The default value is `INFO`.

```python
from gemini_webapi import set_log_level

set_log_level("DEBUG")
```

> [!NOTE]
>
> Calling `set_log_level` for the first time will **globally** remove all existing loguru handlers. You may want to configure logging directly with loguru to avoid this issue and have more advanced control over logging behaviors.
