/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "test_common_macros.hpp"

#include "xtensor/core/xfunction.hpp"
#include "xtensor/containers/xtensor.hpp"

// --------------------------------------------
//  tuple_cat_all : variadic tuple concatenation
// --------------------------------------------
template<class... Tuples>
constexpr auto tuple_cat_all(Tuples&&... tps)
{
    return std::tuple_cat(std::forward<Tuples>(tps)...);
}

// --------------------------------------------
//  Helpers to detect expression vs leaf
// --------------------------------------------

// Detect whether T models xtensor's expression interface
template<class T>
using is_xexpr = std::is_base_of<xt::xexpression<T>, T>;

// Leaf node = actual array/container (not an expression node)
template<class T>
constexpr bool is_leaf_v =
!is_xexpr<std::decay_t<T>>::value
|| xt::detail::is_container<std::decay_t<T>>::value
|| xt::is_xscalar<T>::value;

// --------------------------------------------
//  flatten_impl : main recursive (compile-time) flattener
// --------------------------------------------
template<class E>
constexpr auto flatten_impl(E&& expr);

// flatten for *leaf nodes* (xarray, xtensor, xscalar)
template<class T>
constexpr auto flatten_leaf(T&& leaf)
{
    return std::make_tuple(std::ref(leaf));
}

// flatten for *expression nodes*
template<class D>
constexpr auto flatten_expr(xt::xexpression<D>& expr)
{
    // expr has a tuple of subexpressions: expr.derived_cast().m_e
    auto& e = expr.derived_cast();

    // Recursively flatten every sub-expression using std::apply
    auto children_tuple =
        std::apply(
            [&](auto&... children)
            {
                return tuple_cat_all(flatten_impl(children)...);
            },
            e.arguments()
        );

    // Append this node itself (prefix order)
    return std::tuple_cat(children_tuple, std::make_tuple(std::ref(e)));
}

// dispatch
template<class E>
constexpr auto flatten_impl(E&& expr)
{
    if constexpr (is_leaf_v<E>)
    {
        return flatten_leaf(expr);
    }
    else
    {
        return flatten_expr(expr);
    }
}

// --------------------------------------------
//  Public API: flatten_expression(expr)
// --------------------------------------------
template<class E>
constexpr auto flatten_expression(E&& expr)
{
    return flatten_impl(expr);
}

template<class T>
concept leaf  = is_leaf_v<T>;

template<class T>
concept stem = !is_leaf_v<T>;

template<leaf E>
void step(E&& e)
{
    e.step();
}

template<class E>
void step(E&& /*e*/)
{
    //do nothing
}

template<stem E>
void execute(E&& e)
{
    e.eval();
}

template<class E>
void execute(E&& /*e*/)
{
    //do nothing
}

TEST(xfunction_tuple_wrapper, wrap_func)
{
    xt::xtensor<size_t, 1> a({1,2,3,4});
    xt::xtensor<size_t, 1> b({1,2,3,4});
    xt::xtensor<size_t, 1> result({0,0,0,0});
    auto func = a + b;
    auto flat = flatten_expression(func);

    for (size_t i = 0; i < 4; ++i)
    {
        std::apply([&](auto&&... e) {
            (execute(e), ...);
        }, flat);

        std::apply([&](auto&&... e) {
            (step(e), ...);
         }, flat);
        result.at(i) = std::get<std::tuple_size<decltype(flat)>::value - 1>(flat).get();
    }
}

