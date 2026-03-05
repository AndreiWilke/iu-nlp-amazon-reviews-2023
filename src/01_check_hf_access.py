from datasets import get_dataset_config_names

DATASET = "McAuley-Lab/Amazon-Reviews-2023"

def main():
    print("Dataset:", DATASET)
    configs = get_dataset_config_names(DATASET)
    print("Anzahl configs:", len(configs))
    print("Erste 30 configs:")
    for c in configs[:30]:
        print(" -", c)

if __name__ == "__main__":
    main()
