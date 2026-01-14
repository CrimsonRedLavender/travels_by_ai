from generative.itinerary_generator import generate_itinerary

def main():
    destination = input("Destination: ")
    days = input("Number of days: ")
    budget = input("Budget (low/medium/high): ")
    interests = input("Interests (food, museums, nightlife, nature...): ")

    itinerary = generate_itinerary(destination, days, budget, interests)
    print("\nGenerated Itinerary:\n")
    print(itinerary)

if __name__ == "__main__":
    main()
