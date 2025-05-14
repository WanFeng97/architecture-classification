import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import numpy as np
import torch
from feature_extractor_torch import FeatureExtractor
from classifier import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Here sets the base model name and other parameters, the inpput size should be the same as the one used in training, which is 960 for mobilenet_v3_large. if using other models, input size should be changed accordingly.
base_model_name = "mobilenet_v3_large"
target_size = (224, 224)
style_input_size = 960
style_hidden_size = 1024
style_output_size = 25
architect_input_size = style_input_size + style_output_size
architect_hidden_size = 1024
architect_output_size = 17

# Feature extractor is used to extract the features from the image, and the classifier is used to classify the style and architect.
extractor = FeatureExtractor(base_model_name, target_size=target_size)
style_classifier = MLP(input_size=style_input_size, hidden_size=style_hidden_size,
                        output_size=style_output_size, dropout_rate=0.5).to(device)
style_model_path = "../models/style_classifier.pth"
style_classifier.load_state_dict(torch.load(style_model_path, map_location=device))
style_classifier.eval()

architect_classifier = MLP(input_size=architect_input_size, hidden_size=architect_hidden_size,
                            output_size=architect_output_size, dropout_rate=0.5).to(device)
architect_model_path = "../models/architect_classifier.pth"
architect_classifier.load_state_dict(torch.load(architect_model_path, map_location=device))
architect_classifier.eval()

# Define the labels for styles and architects, since the model is trained on 25 styles and 17 architects, the labels should be the same as the one used in training. Decode the labels to get the names of the styles and architects.
style_labels = {
    0: "Achaemenid architecture", 1: "American Foursquare architecture", 2: "American craftsman style",
    3: "Ancient Egyptian architecture", 4: "Art Deco architecture", 5: "Art Nouveau architecture",
    6: "Baroque architecture", 7: "Bauhaus architecture", 8: "Beaux-Arts architecture",
    9: "Byzantine architecture", 10: "Chicago school architecture", 11: "Colonial architecture",
    12: "Deconstructivism", 13: "Edwardian architecture", 14: "Georgian architecture",
    15: "Gothic architecture", 16: "Greek Revival architecture", 17: "International style",
    18: "Novelty architecture", 19: "Palladian architecture", 20: "Postmodern architecture",
    21: "Queen Anne architecture (Victorian)", 22: "Romanesque architecture",
    23: "Russian Revival architecture", 24: "Tudor Revival architecture"
}

architect_labels = {
    0: "Adamo Boari", 1: "Baldassare Longhena", 2: "Barott and Blackader",
    3: "Coop Himmelb(l)au", 4: "Daniel Libeskind", 5: "Donato Bramante",
    6: "Francisco Salamone", 7: "Frank Gehry", 8: "Imhotep",
    9: "Ludwig Mies van der Rohe", 10: "Nikaure", 11: "Philip Johnson",
    12: "Pierre Le Muet", 13: "Rem Koolhaas", 14: "Sneferu",
    15: "Walter Gropius", 16: "Zaha Hadid"
}

# The descriptions are used to provide more information about the styles and architects, and they are used to show the background and culture of the styles and architects, supporting RO2 and O2 of the project.
style_descriptions = {
    "Achaemenid architecture": "The Achaemenid architectural style emerged during the Persian Empire's rule from 550-330 BCE and demonstrated the multicultural nature of the empire. The Persian architects combined design elements from styles including Mesopotamian, Egyptian and Greek traditions to create an imperial style that distinct from others. The capital city Pasargadae established by Cyrus the Great introduced pavilions with wide spacing and the apadana hall design which became the foundation for Persian architectural development. The construction at Persepolis started under Darius I included stone columns which supported massive wooden roofs through double-headed animal capitals.",
    "American craftsman style": "The American Craftsman style (c. 1900-1930) developed as a domestic architectural movement of the Arts and Crafts movement to fight against declining quality of goods during the Industrial Revolution. The United States began its search for design simplicity and honesty in the early 1900s which led to the development of this style that celebrates natural materials and hand-made construction for middle-class homes. The typical Craftsman house design features low-profile architecture with one or two stories and gabled roofs that have wide eaves and exposed structural elements which showcase construction methods.",
    "American Foursquare architecture": "The American Foursquare architecture was popular between mid-1890s to the late 1930s as a practical American house design which opposed Victorian ornateness while seeking functional family-friendly homes. The basic design of a Foursquare house includes a two-and-a-half-story square structure with a hipped roof and central dormer. The front elevation of this house design includes a complete porch that extends across the entire width while featuring wide staircase steps which create an inviting outdoor living area. The widespread adoption of this design pattern coexisted with American Craftsman Style in early 20th century while maintaining their functional simplicity principles.",
    "Ancient Egyptian architecture": "The monumental building tradition of ancient Egypt spanned over three thousand years until the Roman era. The stable theocratic society of the Nile Valley directed its efforts toward eternity by constructing large tombs and temples which guaranteed pharaohs' afterlives and honoured gods. The construction of homes used sun-baked mud bricks but temples and tombs required stone materials including limestone, sandstone and granite. The discovery of Tutankhamun's tomb triggered an Egyptomania movement that brought Egyptian inspiration into Art Deco architecture during the 19th and early 20th centuries.",
    "Art Deco architecture": "The Art Deco architectural movement emerged during the 1920s and 1930s as a modernist design which combined glamour with sleek geometric form and technological progress. The style became popular in Europe at the 1925 Paris Exposition of Decorative Arts where it introduced a new aesthetic that moved from Art Nouveau's flowing organic designs to Machine Age aesthetics. The main characteristics of Art Deco buildings consist of geometric shapes like ziggurats and setbacks and spires. Although International Style surpassed Art Deco's ornate designs by the late 1930s, Art Deco's combination of modernity with craftsmanship continues to live through.",
    "Art Nouveau architecture": "During the period of 1890-1910 Art Nouveau emerged as a decorative avant-garde style which connected the 19th-century Beaux-Arts/Victorian era to early modernism. The design outcome featured a language of flowing organic shapes which caused facades to curve like whiplash while iron and glass became vine-like, floral or other natural forms with decoration. The architectural style features two main characteristics: asymmetrical and dynamic designs which can be seen in Victor Horta's Hôtel Tassel.",
    "Baroque architecture": "During the late 16th century through the mid-18th century Baroque architecture emerged as a dramatic style which originated in Counter-Reformation Italy before spreading throughout the Europe. The architectural style uses classical elements such as twisted columns, colonnades and domes to create a dramatic effect through excessive decoration. The Baroque architecture served as a foundation which subsequent architectural styles either emulated (Rococo) or transformed into simplified designs (Georgian and Palladian architecture).",
    "Bauhaus architecture": "The Bauhaus School at Germany established Bauhaus architecture as a design movement which united art with craft and technology through functionalist principles between 1919 and 1933. Bauhaus founder Walter Gropius along with his colleagues Mies van der Rohe and Hannes Meyer worked to transform society through design by creating buildings that were functional minimalist and suitable for industrial manufacturing. Bauhaus architecture presents itself through geometric forms that mostly use cubic or rectangular blocks while omitting ornamentation and featuring open floor plans and steel and concrete and glass materials. ",
    "Beaux-Arts architecture": "Beaux-Arts architecture drew upon the principles of French neoclassicism from the late 19th century through the early 20th century. The architectural style also combined Renaissance and Baroque styles to produce designs that maintained formal order while featuring modern materials including iron and steel. The architectural style incorporated Baroque high roofs and Renaissance ornament. The excessive nature of Beaux-Arts architecture triggered a reaction in the 20th century which led to modernist and International Style.",
    "Byzantine architecture": "The Eastern Roman (Byzantine) Empire developed Byzantine architecture between the 4th century and the 15th century by uniting late Roman construction techniques with Christian design elements to form a specific sacred building style. The defining innovation was placing a round dome over a square space using pendentives. The walls and vaults received impressive decoration through gold-ground mosaics of Christ and saints along with marble panelling. Russian Revival style church received its deepest influence from Byzantine architecture.",
    "Chicago school architecture": "The “Chicago School” refers to a group of innovators in late 19th-century Chicago who pioneered the modern skyscraper. This style replaced heavy, load-bearing masonry walls with metal structures, which allowed commercial buildings of unprecedented height and large expanses of glass. The Chicago School's innovations co-evolved with early European modernism and directly influenced the later International Style.",
    "Colonial architecture": "During the colonial periods of various regions, European powers established colonial architecture style. This style emerged from the fusion of European building elements with native materials and environmental conditions. The colonial church and courthouse buildings adopted Georgian symmetry and classical orders but displayed more modest proportions and craftsmanship. In the early decades of the 20th century, U.S. adopted the Georgian-era colonial house as its national symbol during the time of American independence for the Colonial Revival movement.",
    "Deconstructivism": "The architectural movement of Deconstructivism emerged during the late 1980s as a reaction against modernist and postmodern conventions through its adoption of fragmented structures and disorderly forms. Deconstructivist buildings display non-linear designs through their skewed structures which combine sharp angles and distorted forms that appear to collide or float. The 1988 MoMA exhibition “Deconstructivist Architecture” curated by Philip Johnson and Mark Wigley introduced Deconstructivism to the world by showcasing architects Gehry, Daniel Libeskind, Rem Koolhaas, Peter Eisenman and Coop Himmelb(l)au.",
    "Edwardian architecture": "The Edwardian architectural period spanned from King Edward VII's rule (1901–1910) through World War I. During the Edwardian period in the UK and its British design-influenced territories, architects moved away from Victorian style by creating lighter designs with restrained ornamentation. The Edwardian domestic architecture featured houses with basic profiles through lighter wall colours and extensive window openings which replaced the heavy Victorian colour schemes.",
    "Georgian architecture": "The Georgian period in British and colonial architecture spanned from 1714 to 1830. The main features of Georgian architecture include symmetry together with proportion and Classical design principles. The exterior design of Georgian buildings included brick walls with white trim, together with stone quoins and decorative crowns (pediments) above doorways. This style, broadly influenced by Palladio’s Palladian ideals, evolved through early-mid phases into Regency and Greek Revival forms, later inspiring 19th-century Colonial Revival and Neo-Georgian movements.",
    "Gothic architecture": "During the Middle Ages, Western Europe adopted Gothic architecture as its primary building style which featured elevated verticality together with new structural approaches. The structural innovations allowed builders to achieve higher ceilings through the pointed arch system, ribbed vaults and flying buttresses. The combination of these elements enabled Gothic cathedrals to attain historic height levels and feature extensive stained-glass windows.",
    "Greek Revival architecture": "During the late neoclassical period, Greek Revival architecture emerged as a direct response to ancient Greek temple designs. The style features Greek temple fronts as its main design element which includes pedimented gables supported by Greek order columns. The buildings received their appearance from white paint or light-coloured stone materials which aimed to duplicate the Parthenon's marble appearance. The style disappeared from popular use during the mid-19th century when Victorian Gothic and eclectic revivals became dominant.",
    "International style": "During the 1920s and 1930s the International Style emerged as a modern architectural movement. The International Style defines buildings through functionalist principles which eliminate all nonessential ornamentation. The architectural features of this style include flat-roofed box volumes combined with white stucco or glass, and steel façades that lack any decorative elements. The European modernist architects Le Corbusier, together with Walter Gropius and Mies van der Rohe opposed the decorative of past architectural periods by advocating that buildings should express their functional and structural elements directly.",
    "Novelty architecture": "The design approach of novelty architecture creates buildings that adopt unusual shapes achieve visual appeal. A restaurant could take the form of a massive hot dog while a coffee shop appears as a large coffee pot and an office resembles a big basket. The main features of novelty architecture consist of its ability to become a landmark while maintaining functional usability. The architectural style influenced serious architectural discussions about symbolism, as seen in Venturi et al.'s Postmodern analysis in Learning from Las Vegas where they validated the  \"Duck \" building as an architectural communication method. ",
    "Palladian architecture": "The Palladian style emerged from the work and writings of Italian Renaissance architect Andrea Palladio. The architectural elements of Palladian design feature temple-inspired building forms and strict symmetry in floor plans and elevation drawings. The facade often displays orderly arrangements and balanced proportions through central features like Palladian window patterns, which feature large arched windows with smaller rectangular openings.",
    "Postmodern architecture": "Postmodern architecture emerged in the 1960s as architects challenged modernist formalism through their rejection of International Style functionalism. Postmodern architects brought back architectural elements such as colour and historical references to replace the dominant glass box structures of modern cities. Postmodern architecture evolved into different directions such as Deconstructivism and revived classicism during the late 1990s because its initial novelty had faded.",
    "Queen Anne architecture (Victorian)": "Queen Anne architecture refers to either the English architecture during the period of Queen Anne (1702–1714) or the British Queen Anne Revival style that gained popularity in the late 19th and early 20th centuries. A typical Queen Anne house presents asymmetrical architecture. It often features a fine brickwork and multiple rooflines. The Queen Anne's irregular design style lost popularity by 1900 when Edwardian, Colonial Revival and Arts & Crafts homes introduced their simpler design features.",
    "Romanesque architecture": "During the 11th and 12th centuries, Romanesque architecture dominated Europe through its massive construction, thick walls and semicircular arches. The decorative elements in Romanesque architecture includes geometric patterns and animal interlace designs in capitals and mouldings. This style represents the cultural fusion of Roman, Carolingian and Byzantine elements. The Romanesque style emerged as the first unified European architectural movement.",
    "Russian Revival architecture": "Russian Revival architecture was a nationalistic architectural movement in the Russian Empire during the mid-19th to early 20th centuries. The movement emerged simultaneously with the Gothic Revival in Britain in and the Renaissance Revival in Italy. The style drew inspiration from medieval Russian architecture and Byzantine traditions. The exterior of buildings features brick or stone walls which receive decorative treatment through tile work or complex brick patterns. The Russian Revival architectural style merged with Art Nouveau during the early 20th century until the 1917 Revolution interrupted its development.",
    "Tudor Revival architecture": "During the late 19th century Tudor Revival architecture spread across Britain and English-speaking nations. The style reveals itself through its half-timbered exterior walls which display decorative timber framing with stucco or brick infill. Tudor Revival houses feature steeply pitched roofs with multiple gables. Tall chimneys and glass windows with diamond panes also represents the style. Tudor Revival emerged as a design movement because it provided a simpler medieval aesthetic through timber and plaster construction instead of stone tracery. This revival lost popularity during the mid-20th century when Modernism arouses."
}

# The function get_style_predictions is used to get the style predictions from the model, and it is used in the process_image function to get the style predictions from the image.
def get_style_predictions(model, embeddings, device):
    model.eval()
    with torch.no_grad():
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
        logits = model(emb_tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

def process_image(image_path, show_background=False):
    img_array = extractor.preprocess_image(image_path)
    if img_array is None:
        raise ValueError("Could not process image!")
    img_batch = np.expand_dims(img_array, axis=0)
    embedding = extractor.get_embeddings_in_batches(img_batch, batch_size=1)
    style_probs = get_style_predictions(style_classifier, embedding, device)
    # Get the top 3 style predictions, which supports RO2 and O2 of the project. Please refer to the project report for more details.
    top3_style_idxs = np.argsort(style_probs[0])[::-1][:3]
    top3_style_names = [(style_labels.get(idx, "Unknown"), round(style_probs[0][idx] * 100, 2)) for idx in top3_style_idxs]

    arch_input = np.concatenate([embedding, style_probs], axis=1)
    arch_input_tensor = torch.tensor(arch_input, dtype=torch.float32, device=device)
    with torch.no_grad():
        arch_logits = architect_classifier(arch_input_tensor)
        arch_probs = torch.softmax(arch_logits, dim=1).cpu().numpy()[0]
    
    top3_arch_idxs = np.argsort(arch_probs)[::-1][:3]
    top3_arch_names = [(architect_labels.get(idx, "Unknown"), round(arch_probs[idx] * 100, 2)) for idx in top3_arch_idxs]

    background_info = ""
    if show_background:
        background_info = style_descriptions.get(top3_style_names[0][0], "No description available.")

    return top3_style_names, top3_arch_names, background_info

# The GUI application is built using tkinter, and it provides a user-friendly interface for uploading images and displaying predictions. The GUI supports RO2 and O2 of the project. 
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Art Style & Architect Predictor")
        self.geometry("1000x900")
        self.resizable(False, False)
        self.configure(background="#ecf0f1")
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.toolbar = tk.Frame(self, bg="#34495e", width=220)
        self.toolbar.grid(row=0, column=0, sticky="ns")
        self.toolbar.grid_propagate(False)
        self.upload_btn = ttk.Button(self.toolbar, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=20, padx=20, anchor="nw")
        self.show_bg_var = tk.IntVar()
        self.bg_checkbox = ttk.Checkbutton(self.toolbar, text="Show Background", variable=self.show_bg_var)
        self.bg_checkbox.pack(pady=20, padx=20, anchor="nw")
        self.content_frame = ttk.Frame(self, padding="20 20 20 20")
        self.content_frame.grid(row=0, column=1, sticky="nsew")
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(1, weight=1)
        self.content_frame.rowconfigure(3, weight=1)
        self.header_label = ttk.Label(self.content_frame, text="Prediction Results", font=("Segoe UI", 18, "bold"))
        self.header_label.grid(row=0, column=0, pady=10, sticky="w")
        self.result_label = ttk.Label(self.content_frame, text="", wraplength=640, justify="left", font=("Segoe UI", 14))
        self.result_label.grid(row=1, column=0, pady=10, sticky="nw")
        self.image_label = ttk.Label(self.content_frame)
        self.image_label.grid(row=2, column=0, pady=10, sticky="nw")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#ecf0f1")
        self.style.configure("TLabel", background="#ecf0f1", font=("Segoe UI", 14), foreground="#2c3e50")
        self.style.configure("Header.TLabel", background="#ecf0f1", font=("Segoe UI", 18, "bold"), foreground="#2980b9")
        self.style.configure("TButton", font=("Segoe UI", 16, "bold"), padding=10, foreground="#ffffff", background="#85c1e9", width=15)
        self.style.map("TButton", background=[("active", "#5dade2")])
        self.style.configure("TCheckbutton", width=15, font=("Segoe UI", 14, "bold"), padding=10, foreground="#ffffff", background="#85c1e9")

    # The upload_image function.
    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            try:
                style_preds, architect_preds, bg_info = process_image(file_path, show_background=bool(self.show_bg_var.get()))
                style_text = "Style Predictions:\n" + ";\n".join([f"{name} ({prob}%)" for name, prob in style_preds])
                arch_text = "Architect Predictions:\n" + ";\n".join([f"{name} ({prob}%)" for name, prob in architect_preds])
                result_text = f"{style_text}\n\n{arch_text}"
                if self.show_bg_var.get():
                    result_text += "\n\nBackground & Culture:\n"
                    self.result_label.config(font=("Segoe UI", 14))
                    self.result_label_bg = ttk.Label(self.content_frame, text=bg_info, wraplength=640, justify="left", anchor="nw", font=("Segoe UI", 12, "italic"))
                    self.result_label_bg.grid(row=3, column=0, pady=(10, 10), sticky="nsew")
                else:
                    if hasattr(self, 'result_label_bg'):
                        self.result_label_bg.destroy()
                self.result_label.config(text=result_text)
                pil_image = Image.open(file_path)
                # Define a dynamic max width/height based on window size or fixed large size, since the image size varies, it needs to be resized to fit the window.
                max_width, max_height = 200, 200
                img_w, img_h = pil_image.size
                scale = min(max_width / img_w, max_height / img_h)
                new_size = (int(img_w * scale), int(img_h * scale))
                resized_image = pil_image.resize(new_size, Image.ANTIALIAS)

                self.photo = ImageTk.PhotoImage(resized_image)
                self.image_label.config(image=self.photo)
            except Exception as e:
                messagebox.showerror("Error", str(e))


if __name__ == '__main__':
    app = App()
    app.mainloop()
