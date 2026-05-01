import yaml


from first_project.src.get_data import get_data
from first_project.src.run_backtest import run_backtest
from first_project.src.strategy import train_strategy


def main():
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    get_data(cfg)
    train_strategy(cfg)
    run_backtest(cfg)


if __name__ == "__main__":
    main()
