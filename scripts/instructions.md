# scripts instructions

Scripts are thin orchestration layers. They should import Hydra/OmegaConf and
project APIs only as needed, avoid optional integrations at import time, and
delegate real work to package modules once implemented.
