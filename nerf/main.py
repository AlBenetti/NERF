"""Main module for nerf."""
import numpy as np
import pandas as pd


def main():
    """Main function for nerf."""
    random_array = np.random.rand(10, 10)
    random_df = pd.DataFrame(random_array)

    print(random_df.mean())


if __name__ == '__main__':
    main()
