#!/usr/bin/python3
import jetson_inference
import jetson_utils
import argparse

#creates terminal input
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()

#loads and classifies image
img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(opt.network)
class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)

#Prints the animal
print("Image is recognized as " + str(class_desc) + " with " + str(round(confidence*100)) + "% confidence")

# List of not at risk species
not_at_risk = [
    "tench", "goldfish", "cock", "hen", "chickadee", "European fire salamander",
    "common newt", "eft", "bullfrog", "tree frog", "tailed frog", "mud turtle",
    "terrapin", "box turtle", "banded gecko", "common iguana", "American chameleon",
    "whiptail", "agama", "frilled lizard", "alligator lizard", "green lizard",
    "harvestman", "scorpion", "garden spider", "tarantula", "wolf spider",
    "tick", "centipede", "ptarmigan", "quail", "coucal", "bee eater", "hornbill",
    "jacamar", "drake", "goose", "white stork", "black stork", "spoonbill", "flamingo",
    "little blue heron", "American egret", "bittern", "crane", "limpkin",
    "European gallinule", "American coot", "bustard", "ruddy turnstone", "red-backed sandpiper",
    "redshank", "dowitcher", "oystercatcher", "pelican", "albatross",
    "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese spaniel",
    "Maltese dog", "Pekinese", "Shih-Tzu", "Blenheim spaniel", "papillon",
    "toy terrier", "Rhodesian ridgeback", "Afghan hound", "basset", "beagle",
    "bloodhound", "bluetick", "black-and-tan coonhound", "Walker hound",
    "English foxhound", "redbone", "borzoi", "Irish wolfhound", "Italian greyhound",
    "whippet", "Ibizan hound", "Norwegian elkhound", "otterhound", "Saluki",
    "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier", "American Staffordshire terrier",
    "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier",
    "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier",
    "Lakeland terrier", "Sealyham terrier", "Airedale", "cairn", "Australian terrier",
    "Dandie Dinmont", "Boston bull", "miniature schnauzer", "giant schnauzer",
    "standard schnauzer", "Scotch terrier", "Tibetan terrier", "silky terrier",
    "soft-coated wheaten terrier", "West Highland white terrier", "Lhasa", "flat-coated retriever",
    "curly-coated retriever", "golden retriever", "Labrador retriever", "Chesapeake Bay retriever",
    "German short-haired pointer", "vizsla", "English setter", "Irish setter",
    "Gordon setter", "Brittany spaniel", "clumber", "English springer", "Welsh springer spaniel",
    "cocker spaniel", "Sussex spaniel", "Irish water spaniel", "kuvasz", "schipperke",
    "groenendael", "malinois", "briard", "kelpie", "komondor", "Old English sheepdog",
    "Shetland sheepdog", "collie", "Border collie", "Bouvier des Flandres", "Rottweiler",
    "German shepherd", "Doberman", "miniature pinscher", "Greater Swiss Mountain dog",
    "Bernese mountain dog", "Appenzeller", "EntleBucher", "boxer", "bull mastiff",
    "Tibetan mastiff", "French bulldog", "Great Dane", "Saint Bernard", "Eskimo dog",
    "malamute", "Siberian husky", "dalmatian", "affenpinscher", "basenji",
    "pug", "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed",
    "Pomeranian", "chow", "keeshond", "Brabancon griffon", "Pembroke",
    "Cardigan", "toy poodle", "miniature poodle", "standard poodle", "Mexican hairless",
    "timber wolf", "white wolf", "red wolf", "coyote", "dingo", "dhole",
    "red fox", "kit fox", "Arctic fox", "grey fox", "tabby", "tiger cat",
    "Persian cat", "Siamese cat", "Egyptian cat", "cougar", "lynx",
    "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah",
    "brown bear", "American black bear", "ice bear", "sloth bear",
    "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle",
    "long-horned beetle", "leaf beetle", "dung beetle", "rhinoceros beetle",
    "weevil", "fly", "bee", "ant", "grasshopper", "cricket", "walking stick",
    "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
    "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly",
    "sulphur butterfly", "lycaenid", "starfish", "sea urchin", "sea cucumber",
    "wood rabbit", "hare", "Angora", "hamster", "porcupine", "fox squirrel",
    "marmot", "beaver", "guinea pig", "sorrel", "zebra", "hog",
    "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram", "bighorn", "ibex", "hartebeest", "impala", "gazelle",
    "Arabian camel", "llama", "weasel", "mink", "polecat", "black-footed ferret",
    "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas", "baboon",
    "macaque", "langur", "colobus", "proboscis monkey", "marmoset", "capuchin",
    "howler monkey", "titi", "spider monkey", "squirrel monkey", "Madagascar cat",
    "indri", "Indian elephant", "African elephant", "lesser panda", "giant panda",
    "barracouta", "eel", "coho", "rock beauty", "anemone fish",
    "sturgeon", "gar", "lionfish", "puffer", "loggerhead", "leatherback",
    "mud puppy", "chiton", "chambered nautilus", "Dungeness crab",
    "rock crab", "fiddler crab", "king crab", "American lobster", "spiny lobster",
    "crayfish", "hermit crab", "isopod"
]

#List of vulnerable species
vulnerable = [
    "great white shark", "white shark", "man-eater", "man-eating shark",
    "tiger shark", "Galeocerdo cuvieri", "hammerhead", "hammerhead shark",
    "electric ray", "crampfish", "numbfish", "torpedo", "stingray",
    "ostrich", "Struthio camelus", "brambling", "Fringilla montifringilla",
    "goldfinch", "Carduelis carduelis", "house finch", "linnet", "Carpodacus mexicanus",
    "junco", "snowbird", "indigo bunting", "indigo finch", "indigo bird",
    "Passerina cyanea", "robin", "American robin", "Turdus migratorius",
    "bulbul", "jay", "magpie", "water ouzel", "dipper", "kite",
    "bald eagle", "American eagle", "Haliaeetus leucocephalus", "vulture",
    "great grey owl", "great gray owl", "Strix nebulosa", "loggerhead turtle",
    "loggerhead", "Caretta caretta", "leatherback turtle", "leatherback",
    "leathery turtle", "Dermochelys coriacea", "Gila monster", "Heloderma suspectum",
    "Komodo dragon", "Komodo lizard", "dragon lizard", "giant lizard",
    "Varanus komodoensis", "African crocodile", "Nile crocodile",
    "Crocodylus niloticus", "American alligator", "Alligator mississipiensis",
    "triceratops", "horned dinosaur", "ringed seal", "Phoca hispida",
    "harbor seal", "common seal", "Phoca vitulina", "dugong", "Dugong dugon",
    "sea cow", "Steller's sea cow", "Hydrodamalis gigas", "tusker",
    "African elephant", "Loxodonta africana", "Indian elephant", "Elephas maximus",
    "red fox", "Vulpes vulpes", "kit fox", "Vulpes macrotis", "Arctic fox",
    "white fox", "Alopex lagopus", "grey fox", "gray fox", "Urocyon cinereoargenteus",
    "tabby", "tabby cat", "tiger cat", "Siamese cat", "Persian cat", "Angora cat",
    "cougar", "puma", "catamount", "mountain lion", "painter", "panther",
    "Felis concolor", "lynx", "Felix lynx", "leopard", "Panthera pardus",
    "snow leopard", "ounce", "Panthera uncia", "jaguar", "Panthera onca",
    "lion", "king of beasts", "Panthera leo", "tiger", "Panthera tigris",
    "cheetah", "Acinonyx jubatus", "brown bear", "bruin", "Ursus arctos",
    "American black bear", "black bear", "Euarctos americanus", "ice bear",
    "polar bear", "Ursus Maritimus", "thespian", "black-footed ferret", "Mustela nigripes",
    "sea otter", "Enhydra lutris", "dugong", "sea cow", "Steller's sea cow",
    "Hydrodamalis gigas", "tusker", "African elephant", "Loxodonta africana",
    "Indian elephant", "Elephas maximus", "white wolf", "Canis lupus", "red wolf",
    "Canis rufus", "dingo", "Canis dingo", "dhole", "Cuon alpinus", "coyote",
    "Canis latrans", "timber wolf", "grey wolf", "gray wolf", "Canis lupus",
    "mule", "mule deer", "Odocoileus hemionus", "hog", "wild boar", "Sus scrofa",
    "warthog", "Phacochoerus aethiopicus", "hippopotamus", "hippo", "river horse",
    "Hippopotamus amphibius", "ox", "bullock", "Bos taurus", "water buffalo",
    "bison", "buffalo", "Bison bison", "ram", "Aries", "bighorn", "bighorn sheep",
    "cimarron", "Ovis canadensis", "ibex", "Capra ibex", "wild goat", "Capra aegagrus",
    "hartebeest", "impala", "Aepyceros melampus", "gazelle", "Arabian camel", "dromedary",
    "Camelus dromedarius", "llama", "Lama glama", "weasel", "Mustela nivalis",
    "mink", "polecat", "otter", "badger", "sea otter", "Enhydra lutris",
    "three-toed sloth", "Bradypus tridactylus", "orangutan", "orang", "Pongo pygmaeus",
    "gorilla", "Gorilla gorilla", "chimpanzee", "chimp", "Pan troglodytes",
    "gibbon", "Hylobates lar", "siamang", "Symphalangus syndactylus", "guenon",
    "patas", "Erythrocebus patas", "baboon", "Papio", "macaque", "Macaca",
    "langur", "colobus", "proboscis monkey", "Nasalis larvatus", "marmoset",
    "Hapale", "capuchin", "Cebus capucinus", "howler monkey", "howler",
    "Alouatta", "titi", "Callicebus moloch", "spider monkey", "Ateles geoffroyi",
    "squirrel monkey", "Saimiri sciureus", "Madagascar cat", "indri", "Indri indri",
    "barracouta", "barracuda", "Sphyraena barracuda", "eel", "Anguilla anguilla",
    "coho", "coho salmon", "blue jack", "silver salmon", "Oncorhynchus kisutch",
    "rock beauty", "Holocanthus tricolor", "anemone fish", "sturgeon",
    "Acipenser sturio", "gar", "garfish", "garpike", "billfish", "Lepisosteus osseus",
    "lionfish", "turkeyfish", "zebra fish", "pterois volitans", "puffer",
    "pufferfish", "blowfish", "globefish", "loggerhead turtle", "loggerhead",
    "Caretta caretta", "leatherback turtle", "leatherback", "leathery turtle",
    "Dermochelys coriacea", "mud puppy", "chiton", "sea cucumber"
]

#List of endangered species
endangered = [
    "great white shark", "tiger shark", "hammerhead shark", "electric ray",
    "stingray", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "water ouzel",
    "dipper", "kite", "bald eagle", "vulture", "great grey owl", "loggerhead turtle",
    "leatherback turtle", "Gila monster", "Komodo dragon", "African crocodile",
    "American alligator", "triceratops", "ringed seal", "harbor seal", "sea cow",
    "Steller's sea cow", "tusker", "African elephant", "Indian elephant", "red fox",
    "kit fox", "Arctic fox", "grey fox", "tabby", "Siamese cat", "Persian cat",
    "Angora cat", "cougar", "lynx", "leopard", "snow leopard", "jaguar",
    "lion", "tiger", "cheetah", "brown bear", "American black bear", "ice bear",
    "black-footed ferret", "sea otter", "three-toed sloth", "orangutan", "gorilla",
    "chimpanzee", "gibbon", "siamang", "guenon", "patas", "baboon", "macaque",
    "langur", "colobus", "proboscis monkey", "marmoset", "capuchin", "howler monkey",
    "titi", "spider monkey", "squirrel monkey", "Madagascar cat", "indri",
    "Indian elephant", "African elephant", "lesser panda", "giant panda",
    "barracouta", "eel", "coho", "rock beauty", "anemone fish", "sturgeon",
    "gar", "lionfish", "puffer", "loggerhead", "leatherback", "mud puppy", "chiton",
    "lorikeet"
]


#using list comprehension
#checking if string contains list element from each of 3 lists
narTrue = any(ele in class_desc for ele in not_at_risk)
vulTrue = any(ele in class_desc for ele in vulnerable)
endTrue = any(ele in class_desc for ele in endangered)


# print result based on what list the animal is from
if narTrue == True:
    print("This species is not at risk.")
elif vulTrue == True:
    print("This species is vulnerable.")
elif endTrue == True:
    print("This species is endangered.")

#prints result
print("The above statement may be innacurate. A much better source for such information would be www.google.com.")