#ifndef _JSON_PROXY_TEMPLATE_HPP_
#define _JSON_PROXY_TEMPLATE_HPP_
#include <iostream>
#include <array>
#include <exception>
#include <stdexcept>

#if __cplusplus >= 201703L //C++17
template <typename E, E V>
constexpr std::string_view get_enum_value_name()
{
  std::string name{__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__) - 2};
  for (std::size_t i = name.size(); i > 0; --i)
  {
    if (!((name[i - 1] >= '0' && name[i - 1] <= '9') ||
          (name[i - 1] >= 'a' && name[i - 1] <= 'z') ||
          (name[i - 1] >= 'A' && name[i - 1] <= 'Z') ||
          (name[i - 1] == '_')))
    {
      name.remove_prefix(i);
      break;
    }
  }
  if (name.size() > 0 && ((name.front() >= 'a' && name.front() <= 'z') ||
                          (name.front() >= 'A' && name.front() <= 'Z') ||
                          (name.front() == '_')))
  {
    return name;
  }
  return {}; // Invalid name.
}

// 该函数可以分辨
// bool is_valid() [E = Color, V = Color::GREEN]
// bool is_valid() [E = Color, V = 1]
template <typename E, E V>
constexpr bool is_valid()
{
  // get_enum_value_name来自于上面一小节。
  return get_enum_value_name<E, V>().size() != 0;
}

// 制造std::integer_sequence，在[-value,value]之间
template <int... Is>
constexpr auto make_integer_list_wrapper(std::integer_sequence<int, Is...>)
{
  constexpr int half_size = sizeof...(Is) / 2;
  return std::integer_sequence<int, (Is-half_size)...>();
}

// 编译器已知的测试integer sequence
constexpr auto test_integer_sequence_v = make_integer_list_wrapper(std::make_integer_sequence<int, 256>());

template <typename E, int... Is>
constexpr size_t get_enum_size(std::integer_sequence<int, Is...>)
{
  constexpr std::array<bool, sizeof...(Is)> valid{is_valid<E, static_cast<E>(Is)>()...};
  constexpr std::size_t count = [](decltype((valid)) valid_) constexpr noexcept->std::size_t
  {
    auto count_ = std::size_t{0};
    for (std::size_t i_ = 0; i_ < valid_.size(); ++i_)
    {
      if (valid_[i_])
      {
        ++count_;
      }
    }
    return count_;
  }
  (valid);
  return count;
}

// 一个enum class里面有几个值。
template <typename E>
constexpr std::size_t enum_size_v = get_enum_size<E>(test_integer_sequence_v);

template <typename E, int... Is>
constexpr auto get_all_valid_values(std::integer_sequence<int, Is...>)
{
  constexpr std::array<bool, sizeof...(Is)> valid{is_valid<E, static_cast<E>(Is)>()...};
  constexpr std::array<int, sizeof...(Is)> integer_value{Is...};
  std::array<int, enum_size_v<E>> values{};
  for (std::size_t i = 0, v = 0; i < sizeof...(Is); ++i)
  {
    if (valid[i])
    {
      values[v++] = integer_value[i];
    }
  }
  return values;
}

template <typename E, int... Is>
constexpr auto get_all_valid_names(std::integer_sequence<int, Is...>)
{
  constexpr std::array<std::string_view, sizeof...(Is)> names{get_enum_value_name<E, static_cast<E>(Is)>()...};
  std::array<std::string_view, enum_size_v<E>> valid_names{};
  for (std::size_t i = 0, v = 0; i < names.size(); ++i)
  {
    if (names[i].size() != 0)
    {
      valid_names[v++] = names[i];
    }
  }
  return valid_names;
}

template <typename E>
constexpr auto enum_names_v = get_all_valid_names<E>(test_integer_sequence_v);

template <typename E>
constexpr auto enum_values_v = get_all_valid_values<E>(test_integer_sequence_v);

template <typename E>
constexpr E string2enum(const std::string_view str)
{
  constexpr auto valid_names = enum_names_v<E>;
  constexpr auto valid_values = enum_values_v<E>;
  constexpr auto enum_size = enum_size_v<E>;
  for (size_t i = 0; i < enum_size; ++i)
  {
    if (str == valid_names[i])
    {
      return static_cast<E>(valid_values[i]);
    }
  }
  throw std::invalid_argument{"Invale value"};
}

template <typename E>
constexpr std::string enum2string(E V)
{
  constexpr auto valid_names = enum_names_v<E>;
  constexpr auto valid_values = enum_values_v<E>;
  constexpr auto enum_size = enum_size_v<E>;
  for (size_t i = 0; i < enum_size; ++i)
  {
    if (static_cast<int>(V) == valid_values[i])
    {
      return valid_names[i];
    }
  }
  throw std::invalid_argument{"Invale value"};
}

#else

#endif

// 该函数可以分辨
// bool is_valid() [E = Color, V = Color::GREEN]
// bool is_valid() [E = Color, V = 1]



#endif

// enum class Color : int
// {
//   RED = -2,
//   BLUE = 0,
//   GREEN = 2
// };


// int main()
// {
//   const auto pretty_print = [](const std::string &name, const auto &array) {
//     std::cout << name << ": [";
//     for (const auto &value : array)
//     {
//       std::cout << value << ", ";
//     }
//     std::cout << "]" << std::endl;
//   };

//   const auto &valid_names = enum_names_v<Color>;
//   const auto &valid_values = enum_values_v<Color>;
//   pretty_print("Valid Values", valid_values);
//   pretty_print("Valid Names", valid_names);

//   static_assert(string2enum<Color>("RED") == Color::RED);
//   static_assert(string2enum<Color>("BLUE") == Color::BLUE);
//   static_assert(string2enum<Color>("GREEN") == Color::GREEN);
//   // Compile Error, LOL!
//   // static_assert(string2enum<Color>("GRAY") == Color::GREEN);

//   std::cout << "RED: " << static_cast<int>(string2enum<Color>("RED")) << std::endl;
//   std::cout << "BLUE: " << static_cast<int>(string2enum<Color>("BLUE")) << std::endl;
//   std::cout << "GREEN: " << static_cast<int>(string2enum<Color>("GREEN")) << std::endl;
//   // throw exception, LOL!
//   // std::cout << "GREEN: " << static_cast<int>(string2enum<Color>("GRAY")) << std::endl;

//   static_assert(enum2string<Color>(static_cast<Color>(-2)) == "RED");
//   static_assert(enum2string<Color>(static_cast<Color>(0)) == "BLUE");
//   static_assert(enum2string<Color>(static_cast<Color>(2)) == "GREEN");
//   // Compile Error, LOL!
//   // static_assert(enum2string<Color>(static_cast<Color>(4)) == "GRAY");

//   std::cout << "-2 : " << enum2string<Color>(static_cast<Color>(-2)) << std::endl;
//   std::cout << "0 : " << enum2string<Color>(static_cast<Color>(0)) << std::endl;
//   std::cout << "2 : " << enum2string<Color>(static_cast<Color>(2)) << std::endl;
//   // throw exception, LOL!
//   // std::cout << "1 : " << enum2string<Color>(static_cast<Color>(1)) << std::endl;

//   return 0;
// }

