#include "json_encoder.hpp"
#include "json_proxy_template.hpp"

#define GENERATE_ENUM_STRINGS  // Start string generation
#include "ucloud_json_format.hpp" //this time, this file is used as cpp.    
#undef GENERATE_ENUM_STRINGS
// #include "ucloud_json_format.cpp"
    

bool UcloudJsonEncoder::initial_context_with_string(const std::string &jsonStr){
    if (jsonStr.empty() || jsonStr == "") return true;

    auto rawJsonLength = static_cast<int>(jsonStr.length());
    JSONCPP_STRING err;
    Json::Value root;
    const std::unique_ptr<Json::CharReader> reader(_Rbuilder.newCharReader());
    if (!reader->parse(jsonStr.c_str(), jsonStr.c_str() + rawJsonLength, &root, &err)) {
        return false;
    }
    _prevRoot = root;
    return true;
}

bool UcloudJsonEncoder::add_context(tagJSON_ROOT rootNode, tagJSON_ATTR attrNode, const std::string &data){
    if(_roots.find(rootNode) == _roots.end()){
        _roots[rootNode] = Json::Value();
    }
    // _roots[rootNode][getStringUJSON_ATTR[attrNode]] = data;
    _roots[rootNode][GetStringJSON_ATTR(attrNode)] = data;
    // _roots[rootNode][enum2string<UJSON_ATTR>(attrNode)] = data;
    return true;
}

std::string UcloudJsonEncoder::output_to_string(){
    Json::Value root = _prevRoot;
    for(auto &&rootNode: _roots){
        // root[enum2string<UJSONROOT>(rootNode.first)] = rootNode.second;
        // root[getStringUJSONROOT[rootNode.first]] = rootNode.second;
        root[GetStringJSON_ROOT(rootNode.first)] = rootNode.second;
    }

    std::string json_file = Json::writeString(_Wbuilder, root);
    return json_file;
}