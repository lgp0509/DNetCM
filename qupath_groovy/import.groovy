// ----------- IMPORT JSON
import qupath.lib.objects.PathObject
import qupath.lib.io.GsonTools

def gson = GsonTools.getInstance(true)

def json = new File("Z:/LN/output_png/170038 PAS.mrxs.json").text

//println json   "Y:/RESULT/output_png/130705 PAS_TEST.mrxs.json" "E:/DN_Slide/130049 PAS.mrxs.json"
// Read the annotations
def type = new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>() {}.getType()
def deserializedAnnotations = gson.fromJson(json, type)
// Set the annotations to have a different name (so we can identify them) & add to the current image
// deserializedAnnotations.eachWithIndex {annotation, i -> annotation.setName('New annotation ' + (i+1))}   # --- THIS WON"T WORK IN CURRENT VERSION
addObjects(deserializedAnnotations)
