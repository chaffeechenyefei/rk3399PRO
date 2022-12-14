// File name: "EnumToString.h"
#if (!defined(_ENUM_TO_STRING_E_HPP_) || !defined(_ENUM_TO_STRING_S_HPP_))
#ifndef GENERATE_ENUM_STRINGS
    #ifndef _ENUM_TO_STRING_E_HPP_
        #define _ENUM_TO_STRING_E_HPP_
        #undef DECL_ENUM_ELEMENT
        #undef BEGIN_ENUM
        #undef END_ENUM
        #define DECL_ENUM_ELEMENT( element ) element
        #define BEGIN_ENUM( ENUM_NAME ) enum class tag##ENUM_NAME
        #define END_ENUM( ENUM_NAME ) ; static const char* GetString##ENUM_NAME(tag##ENUM_NAME index);
    #endif
#else
    #ifndef _ENUM_TO_STRING_S_HPP_
        #define _ENUM_TO_STRING_S_HPP_
        #undef DECL_ENUM_ELEMENT
        #undef BEGIN_ENUM
        #undef END_ENUM
        #define DECL_ENUM_ELEMENT( element ) #element
        #define BEGIN_ENUM( ENUM_NAME ) static const char* gs_##ENUM_NAME [] =
        #define END_ENUM( ENUM_NAME ) ; const char* GetString##ENUM_NAME(tag##ENUM_NAME index){ return gs_##ENUM_NAME [static_cast<int>(index)]; }
    #endif
#endif
/**
 * https://www.codeproject.com/Articles/10500/Converting-C-enums-to-strings
 * 修改内容:
 * 1. #define BEGIN_ENUM( ENUM_NAME ) typedef enum tag##ENUM_NAME => 
 *    #define BEGIN_ENUM( ENUM_NAME ) enum class tag##ENUM_NAME
 * 2. 增加static声明, 避免multiple definition, 因为虽然函数在.h中声明, 但实质是实现, 所以需要加static
 * 3. #define END_ENUM( ENUM_NAME )不需要后缀了.
 */
/**
 * #表示：对应变量字符串化  
 * ##表示：把宏参数名与宏定义代码序列中的标识符连接在一起，形成一个新的标识符
 * 连接符#@：它将单字符标记符变换为单字符，即加单引号
 */
#endif

