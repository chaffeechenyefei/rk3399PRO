#ifndef _JSON_ENCODER_HPP_
#define _JSON_ENCODER_HPP_
#include "json/json.h"
#include <string>
#include <vector>
#include <map>

#include "ucloud_json_format.hpp"

class UcloudJsonEncoder{
public:
    UcloudJsonEncoder(){};
    ~UcloudJsonEncoder(){};

    bool initial_context_with_string(const std::string &jsonStr);

    bool add_context(tagJSON_ROOT rootNode, tagJSON_ATTR attrNode, const std::string &data);

    std::string output_to_string();
    

protected:
    Json::StreamWriterBuilder _Wbuilder;
    Json::CharReaderBuilder _Rbuilder;
    std::map<tagJSON_ROOT,Json::Value> _roots;
    Json::Value _prevRoot;
    
};

#endif

/**
 * JSON根属性 通过enum class解决重名问题
 */
// enum class UJSONROOT{
//     FACE_ATTRIBUTION,
//     OTHERS,
// };
// static std::map<UJSONROOT,std::string> getStringUJSONROOT = {
//     {UJSONROOT::FACE_ATTRIBUTION, "face_attribution"},
//     {UJSONROOT::OTHERS, "others"},
// };


/**
 * JSON次级属性
 */
// enum class UJSON_ATTR{
//     AGE,
//     SEX,
//     MASK,
//     NOTE,
//     OTHERS,
// };
// static std::map<UJSON_ATTR, std::string> getStringUJSON_ATTR = {
//     {UJSON_ATTR::AGE, "age"},
//     {UJSON_ATTR::SEX, "sex"},
//     {UJSON_ATTR::MASK, "mask"},
//     {UJSON_ATTR::OTHERS, "others"},
//     {UJSON_ATTR::NOTE, "note"},
// };
