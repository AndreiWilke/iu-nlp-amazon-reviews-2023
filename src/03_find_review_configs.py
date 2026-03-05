from datasets import get_dataset_config_names

DATASET = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORIES = ["Electronics", "Home_and_Kitchen", "Clothing_Shoes_and_Jewelry"]

def main():
    configs = get_dataset_config_names(DATASET)
    print("configs total:", len(configs))

    for cat in CATEGORIES:
        # alles, was "review" enthält und die Kategorie im Namen hat
        hits = [c for c in configs if ("review" in c) and (cat in c)]
        print("\n==", cat, "==")
        print("Treffer:", len(hits))
        for h in hits[:60]:
            print(" -", h)

if __name__ == "__main__":
    main()
